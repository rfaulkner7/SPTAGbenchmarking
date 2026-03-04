// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOGGING_H_
#define _SPTAG_HELPER_LOGGING_H_

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

#pragma warning(disable:4996)

namespace SPTAG
{
    namespace Helper
    {
        enum class LogLevel
        {
            LL_Debug = 0,
            LL_Info,
            LL_Status,
            LL_Warning,
            LL_Error,
            LL_Assert,
            LL_Count,
            LL_Empty
        };

        class Logger 
        {
        public:
            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...) = 0;
        };

        class LoggerHolder
        {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 	202002L) || __cplusplus >= 	202002L)
        private:
            std::atomic<std::shared_ptr<Logger>> m_logger;
        public:
            LoggerHolder(std::shared_ptr<Logger> logger) : m_logger(logger) {}

            void SetLogger(std::shared_ptr<Logger> p_logger)
            {
                m_logger = p_logger;
            }

            std::shared_ptr<Logger> GetLogger()
            {
                return m_logger;
            }
#else
        private:
            std::shared_ptr<Logger> m_logger;
        public:
            LoggerHolder(std::shared_ptr<Logger> logger) : m_logger(logger) {}

            void SetLogger(std::shared_ptr<Logger> p_logger)
            {
                std::atomic_store(&m_logger, p_logger);
            }

            std::shared_ptr<Logger> GetLogger()
            {
                return std::atomic_load(&m_logger);
            }
#endif
        };


        class SimpleLogger : public Logger {
        public:
            SimpleLogger(LogLevel level, bool includeTimestamp = false) 
                : m_level(level), m_includeTimestamp(includeTimestamp) {}

            static std::string GetTimestamp()
            {
                auto now = std::chrono::system_clock::now();
                auto time_t_now = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;
                
                std::tm tm_now;
#ifdef _MSC_VER
                localtime_s(&tm_now, &time_t_now);
#else
                localtime_r(&time_t_now, &tm_now);
#endif
                std::ostringstream oss;
                oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S")
                    << '.' << std::setfill('0') << std::setw(3) << ms.count();
                return oss.str();
            }

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level) return;

                if (m_includeTimestamp)
                {
                    printf("[%s] ", GetTimestamp().c_str());
                }
                if (level != LogLevel::LL_Empty) printf("[%d] ", (int)level);

                va_list args;
                va_start(args, format);
                
                vprintf(format, args);
                fflush(stdout);

                va_end(args);
            }

            void SetTimestamp(bool enable) { m_includeTimestamp = enable; }
        private:
            LogLevel m_level;
            bool m_includeTimestamp;
        };

        class FileLogger : public Logger {
        public:
            FileLogger(LogLevel level, const char* file, bool includeTimestamp = false) 
                : m_level(level), m_includeTimestamp(includeTimestamp)
            {
                m_handle.reset(new std::fstream(file, std::ios::out));
            }

            ~FileLogger()
            {
                if (m_handle != nullptr) m_handle->close();
            }

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level || m_handle == nullptr || !m_handle->is_open()) return;

                va_list args;
                va_start(args, format);

                char buffer[1024];
                int ret = vsprintf(buffer, format, args);
                if (ret > 0)
                {
                    if (m_includeTimestamp)
                    {
                        std::string timestamp = "[" + SimpleLogger::GetTimestamp() + "] ";
                        m_handle->write(timestamp.c_str(), timestamp.size());
                    }
                    m_handle->write(buffer, strlen(buffer));
                }
                else
                {
                    std::string msg("Buffer size is not enough!\n");
                    m_handle->write(msg.c_str(), msg.size());
                }

                m_handle->flush();
                va_end(args);
            }

            void SetTimestamp(bool enable) { m_includeTimestamp = enable; }
        private:
            LogLevel m_level;
            bool m_includeTimestamp;
            std::unique_ptr<std::fstream> m_handle;
        };

        // TeeLogger: outputs to both console and file with timestamps
        class TeeLogger : public Logger {
        public:
            TeeLogger(LogLevel level, const char* file, bool includeTimestamp = true) 
                : m_level(level), m_includeTimestamp(includeTimestamp)
            {
                m_handle.reset(new std::fstream(file, std::ios::out));
            }

            ~TeeLogger()
            {
                if (m_handle != nullptr) m_handle->close();
            }

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level) return;

                va_list args;
                va_start(args, format);

                char buffer[1024];
                int ret = vsprintf(buffer, format, args);
                
                std::string timestamp = m_includeTimestamp ? "[" + SimpleLogger::GetTimestamp() + "] " : "";
                std::string levelStr = (level != LogLevel::LL_Empty) ? "[" + std::to_string((int)level) + "] " : "";
                
                if (ret > 0)
                {
                    // Output to console
                    printf("%s%s%s", timestamp.c_str(), levelStr.c_str(), buffer);
                    fflush(stdout);
                    
                    // Output to file
                    if (m_handle != nullptr && m_handle->is_open())
                    {
                        m_handle->write(timestamp.c_str(), timestamp.size());
                        m_handle->write(levelStr.c_str(), levelStr.size());
                        m_handle->write(buffer, strlen(buffer));
                        m_handle->flush();
                    }
                }
                else
                {
                    std::string msg("Buffer size is not enough!\n");
                    printf("%s", msg.c_str());
                    if (m_handle != nullptr && m_handle->is_open())
                    {
                        m_handle->write(msg.c_str(), msg.size());
                        m_handle->flush();
                    }
                }

                va_end(args);
            }

            void SetTimestamp(bool enable) { m_includeTimestamp = enable; }
        private:
            LogLevel m_level;
            bool m_includeTimestamp;
            std::unique_ptr<std::fstream> m_handle;
        };
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_LOGGING_H_
