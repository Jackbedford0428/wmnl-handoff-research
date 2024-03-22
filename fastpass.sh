#!/usr/bin/env bash

if [ "$1" == "-a" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/android
elif [ "$1" == "-m" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/mobileinsight
elif [ "$1" == "-i" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/iperf
elif [ "$1" == "-u" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/udp-socket-programming/v3
elif [ "$1" == "-t" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/tcp-socket-programming/v3
elif [ "$1" == "-q" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/quectel
elif [ "$1" == "-g" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/gps
elif [ "$1" == "-s" ]; then
  cd ./jackbedford/wmnl-handoff-research/experimental-tools/sync
else
  cd ./jackbedford/wmnl-handoff-research/experimental-tools
fi
