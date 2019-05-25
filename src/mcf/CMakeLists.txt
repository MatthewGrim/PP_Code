##
#  CMake script for the MCF code:
##

# Set the name of the project and target:
SET(TARGET "mcf")

# Declare all source files the target consists of.
FILE(GLOB_RECURSE TARGET_SRC  "./*")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

# Link to GTEST
include_directories(../)
#include_directories(../ /usr/local/include/)
link_directories(/usr/local/lib)

# Find Deal II
FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
