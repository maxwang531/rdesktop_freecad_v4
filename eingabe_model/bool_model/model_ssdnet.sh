#!/bin/bash
cd /config/eingabe_model/bool_model/
for filename in *.stl; do
./binvox -e -d 64 "$filename"
done

