// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <boost/program_options.hpp>
#include <omp.h>

#ifndef _WINDOWS
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#else
#error "search_server is currently implemented for POSIX platforms only."
#endif

#include "linux_aligned_file_reader.h"
#include "pq_flash_index.h"

namespace po = boost::program_options;

namespace
{

#pragma pack(push, 1)
struct RequestHeader
{
    uint32_t query_id;
    uint32_t k;
    uint32_t l;
    float et_theta;
};

struct ResponseHeader
{
    uint32_t query_id;
    uint64_t server_us;
};
#pragma pack(pop)

static_assert(sizeof(RequestHeader) == 16, "Unexpected request header size");
static_assert(sizeof(ResponseHeader) == 12, "Unexpected response header size");

bool recv_all(int fd, void *buffer, size_t length)
{
    auto *ptr = static_cast<std::uint8_t *>(buffer);
    size_t received = 0;
    while (received < length)
    {
        ssize_t rc = ::recv(fd, ptr + received, length - received, 0);
        if (rc == 0)
        {
            return false;
        }
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        received += static_cast<size_t>(rc);
    }
    return true;
}

bool send_all(int fd, const void *buffer, size_t length)
{
    const auto *ptr = static_cast<const std::uint8_t *>(buffer);
    size_t sent = 0;
    while (sent < length)
    {
#ifdef MSG_NOSIGNAL
        ssize_t rc = ::send(fd, ptr + sent, length - sent, MSG_NOSIGNAL);
#else
        ssize_t rc = ::send(fd, ptr + sent, length - sent, 0);
#endif
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        sent += static_cast<size_t>(rc);
    }
    return true;
}

template <typename T> T cast_query_value(float value)
{
    return static_cast<T>(value);
}

template <> int8_t cast_query_value<int8_t>(float value)
{
    value = std::max(-128.0f, std::min(127.0f, value));
    return static_cast<int8_t>(value);
}

template <> uint8_t cast_query_value<uint8_t>(float value)
{
    value = std::max(0.0f, std::min(255.0f, value));
    return static_cast<uint8_t>(value);
}

template <typename T> class SearchServer
{
  public:
    SearchServer(const std::string &index_prefix, const diskann::Metric metric, const uint32_t num_nodes_to_cache,
                 const uint32_t num_threads, const uint32_t beamwidth)
        : _beamwidth(beamwidth)
    {
#ifdef _WINDOWS
        static_assert(false, "search_server is currently implemented for POSIX platforms only.");
#else
        _reader = std::shared_ptr<AlignedFileReader>(new LinuxAlignedFileReader());
#endif

        _index = std::unique_ptr<diskann::PQFlashIndex<T>>(new diskann::PQFlashIndex<T>(_reader, metric));
        const int rc = _index->load(num_threads, index_prefix.c_str());
        if (rc != 0)
        {
            throw std::runtime_error("Unable to load index, status code " + std::to_string(rc));
        }

        std::vector<uint32_t> node_list;
        std::cout << "Caching " << num_nodes_to_cache << " BFS nodes around medoid(s)" << std::endl;
        _index->cache_bfs_levels(num_nodes_to_cache, node_list);
        _index->load_cache_list(node_list);
        _dimensions = _index->get_data_dim();
        omp_set_num_threads(num_threads);
    }

    void serve(const uint16_t port, const uint32_t worker_threads)
    {
        if (port == 0)
        {
            throw std::invalid_argument("port must be non-zero");
        }
        if (worker_threads == 0)
        {
            throw std::invalid_argument("num_threads must be non-zero");
        }

        const int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0)
        {
            throw std::runtime_error(std::string("socket() failed: ") + std::strerror(errno));
        }

        auto close_listen = [&]() {
            if (listen_fd >= 0)
            {
                ::close(listen_fd);
            }
        };

        int reuse = 1;
        if (::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0)
        {
            close_listen();
            throw std::runtime_error(std::string("setsockopt(SO_REUSEADDR) failed: ") + std::strerror(errno));
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);

        if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
        {
            close_listen();
            throw std::runtime_error(std::string("bind() failed: ") + std::strerror(errno));
        }
        if (::listen(listen_fd, static_cast<int>(worker_threads * 16)) < 0)
        {
            close_listen();
            throw std::runtime_error(std::string("listen() failed: ") + std::strerror(errno));
        }

        std::cout << "Listening on port " << port << " with " << worker_threads << " worker threads" << std::endl;

        std::vector<std::thread> workers;
        workers.reserve(worker_threads);
        for (uint32_t i = 0; i < worker_threads; ++i)
        {
            workers.emplace_back([this]() { this->worker_loop(); });
        }

        while (!_shutdown.load())
        {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            const int client_fd = ::accept(listen_fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
            if (client_fd < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                _shutdown.store(true);
                _queue_cv.notify_all();
                close_listen();
                for (auto &worker : workers)
                {
                    if (worker.joinable())
                    {
                        worker.join();
                    }
                }
                throw std::runtime_error(std::string("accept() failed: ") + std::strerror(errno));
            }

            {
                std::lock_guard<std::mutex> lock(_queue_mutex);
                _fd_queue.push(client_fd);
            }
            _queue_cv.notify_one();
        }

        close_listen();
        _queue_cv.notify_all();
        for (auto &worker : workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

  private:
    void worker_loop()
    {
        while (true)
        {
            int client_fd = -1;
            {
                std::unique_lock<std::mutex> lock(_queue_mutex);
                _queue_cv.wait(lock, [this]() { return _shutdown.load() || !_fd_queue.empty(); });
                if (_shutdown.load() && _fd_queue.empty())
                {
                    return;
                }
                client_fd = _fd_queue.front();
                _fd_queue.pop();
            }

            try
            {
                handle_client(client_fd);
            }
            catch (const std::exception &ex)
            {
                std::cerr << "search_server: request handling failed: " << ex.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "search_server: request handling failed with unknown error" << std::endl;
            }

            ::close(client_fd);
        }
    }

    void handle_client(const int client_fd)
    {
        const auto start = std::chrono::steady_clock::now();

        RequestHeader request{};
        if (!recv_all(client_fd, &request, sizeof(request)))
        {
            throw std::runtime_error("failed to read request header");
        }
        if (request.k == 0 || request.l == 0)
        {
            throw std::runtime_error("invalid request parameters");
        }
        if (request.k > request.l)
        {
            throw std::runtime_error("request K must be <= L");
        }

        std::vector<float> query_f(_dimensions);
        const size_t query_bytes = static_cast<size_t>(_dimensions) * sizeof(float);
        if (!recv_all(client_fd, query_f.data(), query_bytes))
        {
            throw std::runtime_error("failed to read query vector");
        }

        std::vector<T> query(_dimensions);
        for (uint64_t i = 0; i < _dimensions; ++i)
        {
            query[i] = cast_query_value<T>(query_f[i]);
        }

        std::vector<uint64_t> result_ids(request.k);
        std::vector<float> result_dists(request.k);

        if (request.et_theta > 0.0f)
        {
            _index->cached_beam_search(query.data(), request.k, request.l, result_ids.data(), result_dists.data(),
                                       _beamwidth, request.et_theta);
        }
        else
        {
            _index->cached_beam_search(query.data(), request.k, request.l, result_ids.data(), result_dists.data(),
                                       _beamwidth);
        }

        const auto end = std::chrono::steady_clock::now();
        const uint64_t server_us =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        ResponseHeader response_header{request.query_id, server_us};
        const size_t response_size = sizeof(response_header) + request.k * sizeof(uint64_t) + request.k * sizeof(float);
        std::vector<std::uint8_t> response(response_size);

        size_t offset = 0;
        std::memcpy(response.data() + offset, &response_header, sizeof(response_header));
        offset += sizeof(response_header);
        std::memcpy(response.data() + offset, result_ids.data(), request.k * sizeof(uint64_t));
        offset += request.k * sizeof(uint64_t);
        std::memcpy(response.data() + offset, result_dists.data(), request.k * sizeof(float));

        if (!send_all(client_fd, response.data(), response.size()))
        {
            throw std::runtime_error("failed to send response");
        }
    }

    std::shared_ptr<AlignedFileReader> _reader;
    std::unique_ptr<diskann::PQFlashIndex<T>> _index;
    uint64_t _dimensions = 0;
    uint32_t _beamwidth = 4;

    std::queue<int> _fd_queue;
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    std::atomic<bool> _shutdown{false};
};

template <typename T>
int run_search_server(const std::string &index_path_prefix, const diskann::Metric metric,
                      const uint32_t num_nodes_to_cache, const uint32_t num_threads, const uint32_t beamwidth,
                      const uint16_t port)
{
    SearchServer<T> server(index_path_prefix, metric, num_nodes_to_cache, num_threads, beamwidth);
    server.serve(port, num_threads);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    std::string data_type;
    std::string dist_fn;
    std::string index_path_prefix;
    uint16_t port = 9001;
    uint32_t num_threads = static_cast<uint32_t>(omp_get_num_procs());
    uint32_t num_nodes_to_cache = 0;
    uint32_t beamwidth = 4;

    po::options_description desc{"Arguments"};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->default_value("l2"),
                           "distance function <l2/mips>");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix for loading index file components");
        desc.add_options()("port", po::value<uint16_t>(&port)->default_value(9001), "TCP port to listen on");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of worker threads");
        desc.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                           "Number of nodes to cache during search");
        desc.add_options()("beamwidth", po::value<uint32_t>(&beamwidth)->default_value(4),
                           "Beamwidth used for each search");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == "l2")
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == "mips")
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else
    {
        std::cerr << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    try
    {
        if (data_type == "float")
        {
            return run_search_server<float>(index_path_prefix, metric, num_nodes_to_cache, num_threads, beamwidth,
                                            port);
        }
        if (data_type == "int8")
        {
            return run_search_server<int8_t>(index_path_prefix, metric, num_nodes_to_cache, num_threads, beamwidth,
                                             port);
        }
        if (data_type == "uint8")
        {
            return run_search_server<uint8_t>(index_path_prefix, metric, num_nodes_to_cache, num_threads, beamwidth,
                                               port);
        }

        std::cerr << "Unsupported data type " << data_type << std::endl;
        return -1;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "search_server failed: " << ex.what() << std::endl;
        return -1;
    }
}
