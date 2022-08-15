import pandas as pd
import numpy as np
import glob

# % Performs quality assurance on SMS data. Checks for invalid trials (missed taps, late (or
# % early) taps). Determines whether the trial should be rejected by checking 
# % for the presence of consequtive missing taps, or if the number of missed 
# % taps is equal to or greater than MissedCrit. Interpolates invalid taps if
# % the trial has not been rejected.

# TODO: this will have to be accomodated for the data that Francesca gives us

# method for getting ITI, IOI and asynchronies from onsets
def parse_data(onsets1, onsets2):
    iti = np.diff(onsets1)
    ioi = np.diff(onsets2)

    iti = np.insert(iti, 0, 0)
    ioi = np.insert(ioi, 0, 0)

    asyn = np.subtract(onsets1, onsets2)

    return iti, ioi, asyn

# TODO: set the first values to 0 (i.e. start from that)
# this needs to be discussed
# TODO: should they be matrices of 2 vectors instead?
def qa_data(missed_taps, asyn, ITI, IOI, TapTimes, MissedCrit, TestRange):
    # for easier writing
    start, end = TestRange
    iti = ITI

    # check if there are any asynchronies too big
    bound = IOI * 0.5
    large_asyn = np.array(asyn) < -1 * np.array(bound)
    invalid = np.logical_or(missed_taps[start : end], large_asyn[start : end])

    n_missed_taps = sum(missed_taps[start : end])
    n_large_asyn = sum(large_asyn[start : end])
    n_invalid = sum(invalid)

    print("Checking data quality")
    print("{} large asynchronies within Test Range".format(n_large_asyn))
    print("{} missed taps within Test Range".format(n_missed_taps))
    print("{} total invalid rows".format(n_invalid))


    if n_invalid > MissedCrit:
        print("Data rejected. Reason: too many invalid rows.")
        return TapTimes, ITI, asyn

    for i in range(len(invalid) - 1):
        if invalid[i] and invalid[i + 1]:
            print("Data rejected. Reason: data has consecutive invalid rows.")
            return TapTimes, ITI, asyn
    
    # TODO: does not check if it is start or end, will break
    # need to talk about how interpolation is done then
    if n_invalid > 0:
        print("Interpolating missing data using average")

        invalid_indx = [i for i, x in enumerate(invalid) if x]

        for index in invalid_indx:
            asyn[index + start] = (asyn[index + start - 1] + asyn[index + start + 1]) / 2
            TapTimes[index + start] = (TapTimes[index + start - 1] + TapTimes[index + start + 1]) / 2

            print("Interpolated trial {}".format(index + start + 1))
        
        iti = np.diff(TapTimes)
        iti = np.insert(iti, 0, 0)

    return TapTimes, iti, asyn



# TODO: this will not exist in the final version
filename = glob.glob("*Stable*.txt")[0]

data = pd.read_csv(filename, header=None)

onsets1, onsets2 = data[3].tolist(), data[7].tolist()

# get the velocities, AND them (True = both hit, False = at least one missed) and then inverse result
# TODO: this needs to be changed to accomodate the fact that both ITI and IOI are being checked
# (so should be split into two columns)
# and it might need to be moved to the function itself (depends on the data format)
missed_taps = np.logical_not(np.logical_and(data[5].tolist(), data[9].tolist()))
test_range = [21, 71]

iti, ioi, asyn = parse_data(onsets1, onsets2)

onsets, iti, asyn = qa_data(missed_taps, asyn, iti, ioi, onsets1, 4, test_range)

quality_data = pd.DataFrame(list(zip(iti[21:], asyn[21:], ioi[21:])))

quality_data.to_csv("cleaned_data.csv", header=None, index=None)