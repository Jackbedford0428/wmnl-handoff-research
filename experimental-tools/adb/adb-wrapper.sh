#!/usr/bin/env bash

# https://gist.github.com/wuyexiong/2e73975f6a98dccbca93

# Command Usage:
# ./adb-wrapper.sh ADB_COMMAND
# ./adb-wrapper.sh shell
# 1: R5CRA2EGJ5X  sm08    device
# 2: R5CRA1ET22M  sm07    device

# Run adb devices once, in event adb hasn't been started yet
BLAH=$(adb devices)
# echo $BLAH

# Grab the IDs of all the connected devices / emulators
IDS=($(adb devices | sed '1,1d' | sed '$d' | cut -f 1))
STATUS=($(adb devices | sed '1,1d' | sed '$d' | cut -f 2))
NUMIDS=${#IDS[@]}

# echo ${IDS[@]}
# echo ${STATUS[@]}
# echo $NUMIDS

# declare -a my_array
# my_array=(${my_array[@]} foo)
# my_array=(${my_array[@]} play)
# my_array=(${my_array[@]} toy)
# echo ${my_array[@]}

declare -a DEVICES
for (( x=0; x < $NUMIDS; x++ )); do
    SERIAL=${IDS[x]}
    case $SERIAL in
        "R5CR20FDXHK") DEV="sm00"
            ;;
        "R5CR30P9Z8Y") DEV="sm01"
            ;;
        "R5CRA1GCHFV") DEV="sm02"
            ;;
        "R5CRA1JYYQJ") DEV="sm03"
            ;;
        "R5CRA1EV0XH") DEV="sm04"
            ;;
        "R5CRA1GBLAZ") DEV="sm05"
            ;;
        "R5CRA1ESYWM") DEV="sm06"
            ;;
        "R5CRA1ET22M") DEV="sm07"
            ;;
        "R5CRA1D23QK") DEV="sm08"
            ;;
        "R5CRA2EGJ5X") DEV="sm09"
            ;;
        "R5CRA1ET5KB") DEV="sm10"
            ;;
        "R5CRA1D2MRJ") DEV="sm11"
            ;;
        "73e11a9f") DEV="xm00"
            ;;
        "491d5141") DEV="xm01"
            ;;
        "790fc81d") DEV="xm02"
            ;;
        "e2df293a") DEV="xm03"
            ;;
        "28636990") DEV="xm04"
            ;;
        "f8fe6582") DEV="xm05"
            ;;
        "d74749ee") DEV="xm06"
            ;;
        "10599c8d") DEV="xm07"
            ;;
        "57f67f91") DEV="xm08"
            ;;
        "232145e8") DEV="xm09"
            ;;
        "70e87dd6") DEV="xm10"
            ;;
        "df7aeaf8") DEV="xm11"
            ;;
        "e8c1eff5") DEV="xm12"
            ;;
        "ec32dc1e") DEV="xm13"
            ;;
        "2aad1ac6") DEV="xm14"
            ;;
        "64545f94") DEV="xm15"
            ;;
        "613a273a") DEV="xm16"
            ;;
        "fe3df56f") DEV="xm17"
            ;;
        *) DEV="Unknown"
    esac
    DEVICES=(${DEVICES[@]} $DEV)
done
# echo ${DEVICES[@]}

# Set the default command as "adb shell"
if [[ -z $@ ]]; then  # Check If a Variable Is Empty
    CMD="shell"
else
    CMD=$@
fi

# Check for number of connected devices / emulators
if [[ 0 -eq "$NUMIDS" ]]; then
    # No IDs, exit
    echo "No emulators or devices detected - nothing to do."
    exit 0;
elif [[ 1 -eq "$NUMIDS" ]]; then
    # Just one device / emulator
    adb $CMD
    exit 0;
fi

# If we got here, there are multiple devices, need to get information then prompt user for which device/emulator to uninstall from

# # Grab the model name for each device / emulator
# declare -a MODEL_NAMES
# for (( x=0; x < $NUMIDS; x++ )); do
#     MODEL_NAMES[x]=$(adb devices | grep ${IDS[$x]} | cut -f 1 | xargs -I $ adb -s $ shell cat /system/build.prop | grep "ro.product.model" | cut -d "=" -f 2 | tr -d ' \r\t\n')
# done

# # Grab the platform version for each device / emulator
# declare -a PLATFORM_VERSIONS
# for (( x=0; x < $NUMIDS; x++ )); do
#     PLATFORM_VERSIONS[x]=$(adb devices | grep ${IDS[$x]} | cut -f 1 | xargs -I $ adb -s $ shell cat /system/build.prop | grep "ro.build.version.release" | cut -d "=" -f 2 | tr -d ' \r\t\n')
# done

echo "Multiple devices detected, please select one by a number"
for (( x=0; x < $NUMIDS; x++ )); do
    echo -e "$[x+1]: ${IDS[x]}\t${DEVICES[x]}\t${STATUS[x]}"
done
echo -n "> "
read USER_CHOICE

# Validate user entered a number
if [[ $USER_CHOICE =~ ^[0-9]+$ ]] && [[ $USER_CHOICE -gt 0 ]] && [[ $USER_CHOICE -le $NUMIDS ]]; then
    echo "Executing the following command:"
    echo "    adb -s ${IDS[$USER_CHOICE-1]} $CMD"
    adb -s ${IDS[$USER_CHOICE-1]} $CMD
else
    echo "You must enter a valid number"
fi
