#!/usr/bin/env bash

set -euo pipefail

main() {
    case "$#" in
        "1" )
            if [ "$1" == "v" ]
            then
                local v="--verbose"
            else
                local v=""
            fi
            ;;
        "0" )
            local v=""
            ;;
        * )
            echo "you broke it"
            ;;
    esac
    for imgfile in $(ls images)
    do
        echo "Processing ${imgfile}"
        ./main.py "images/${imgfile}" "output/${imgfile}" $v
    done
}

main "$@"
