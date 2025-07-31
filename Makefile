all: flash

TARGET := src/main

TARGET_MCU?=CH32V003

EXTRA_CFLAGS := -Isrc

include src/ch32fun/ch32fun/ch32fun.mk

# Flash the firmware to the device
flash : cv_flash

# Clean up build files
clean : cv_clean