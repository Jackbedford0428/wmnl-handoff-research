# Mobile Insight: Samsung A42 using laptop
`monitor-example.py`
`iperf_server_single.py`
`iperf_client_single.py`

## Samsung: Diagnostic Mode
Enter diagnostic mode

    $ adb shell
    $ adb devices
    $ adb -s SERIAL_NUMBER shell
    $ setprop sys.usb.config diag,serial_cdev,rmnet,adb
> Exit adb shell automatically, and cannot enter adb shell!

> Cannot share Internet with laptop!

Let your laptop find the USB port

    $ sudo rmmod usbserial
> You may need to rmmod other drivers such as "option", "usb_wwan" first.

    $ sudo modprobe usbserial vendor=0x05c6 product=0x9091
> Now, you can enter adb shell again!

    $ ls /dev/ttyUSB*
> You should see ttyUSB0 tyUSB1 tyUSB2, 0 is for QxDM (i.e., MobileInsight).

> You should see ttyUSB3 tyUSB4 tyUSB5, 3 is for QxDM if using 2 cellphone.

Some useful commands

    $ lsusb
    $ lsusb -t
    $ dmesg
    $ dmesg | grep ttyUSB
    $ getprop sys.usb.config  # in adb shell

## Experiment Setup
#### Server: PC
    $ python3 iperf_server_single.py -d sm01 sm02

#### Client: Laptop - Terminal 1, 2
    $ sudo python3 monitor-example.py /dev/ttyUSB0 9600 sm01
    $ sudo python3 monitor-example.py /dev/ttyUSB3 9600 sm02

#### Client: Laptop - Terminal 3, 4
    $ adb -s PHONE1_SERIAL shell
    $ su
    # python3 iperf_client_single.py -d sm01 -u

    $ adb -s PHONE2_SERIAL shell
    $ su
    # python3 iperf_client_single.py -d sm02 -u

***See Command Usage in python script to understand more!***