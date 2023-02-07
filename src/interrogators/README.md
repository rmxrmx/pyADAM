# Interrogators
This directory contains two "interrogators", scripts made to generate sensorimotor synchronization (SMS) data according to some ADAM parameters. Note that, for both of these scripts, the models you set to use are important, as the script will only use the parts of ADAM that are estimated by those models. If you want to use all of the ADAM parameters, simply name the model something other than the two given options, such as `"generic"`.

N.B.: the interrogators do not produce any files by default. To do that, uncomment lines saving their results to `.csv` files. Make sure that you output the data you need.

## Offline interrogator
`offline_interrogator.py` requires a file of onsets (`seq2.csv` in the original script) to use as the "other" participant. It then can generate onsets by setting ADAM parameters. The strength of this approach is that it is easy to adapt this script to use different combinations of parameters (as seen in the example) to generate different onsets. It is also easy to run this script multiple times, to create more data for your task.

## Online interrogator
`online_interrogator.py`, on the other hand, requires real-time input. The implementation of this is up to the user; the example simply waits a set amount of time (600ms) to simulate the other participant's IOI. The strength of this approach is that there is no theoretical limit for how long the run can be; you can set it to run for either a certain amount of inputs, some time, or whatever else you choose (this is set by writing a condition in the `while` loop). It also allows you to incorporate this script into a real-time robot or computer responding to a participant.