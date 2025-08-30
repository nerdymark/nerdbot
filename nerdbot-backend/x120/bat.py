#!/usr/bin/env python3
#This python script is only suitable for UPS Shield X1200, X1201 and X1202

import struct
import smbus


bus = smbus.SMBus(1)


def readVoltage():
    """
    Read the voltage from the UPS Shield X1200, X1201 and X1202
    """
    address = 0x36
    read = bus.read_word_data(address, 2)
    swapped = struct.unpack("<H", struct.pack(">H", read))[0]
    voltage = swapped * 1.25 /1000/16
    return voltage


def readCapacity():
    """
    Read the capacity from the UPS Shield X1200, X1201 and X1202
    """
    address = 0x36
    read = bus.read_word_data(address, 4)
    swapped = struct.unpack("<H", struct.pack(">H", read))[0]
    capacity = swapped/256
    return capacity
