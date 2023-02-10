#!/bin/bash
for job in {0..159}
do
    echo "Running job ${job}."
    python3 khsic_approach_bike_sharing_data.py ${job}
done



