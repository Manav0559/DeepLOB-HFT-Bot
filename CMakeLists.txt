cmake_minimum_required(VERSION 3.10)
project(HFT_ML_Bot)

set(CMAKE_CXX_STANDARD 17)

# 1. Setup Homebrew paths
include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

# 2. Find the Libraries
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

# 3. Create the executable
add_executable(hft_bot bot.cpp)

# 5. GLUE EVERYTHING TOGETHER (Link Libraries)
target_link_libraries(hft_bot
    Threads::Threads
    CURL::libcurl
    zmq
)