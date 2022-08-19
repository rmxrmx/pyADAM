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


# TODO: since missed_taps is optional, two_participants should be coded via TapTimes,
# where it will be ITI - IOI - Metronome
# if missed_taps is True, assume that taps with onset = 0 are missed (except the 1st one)
# TODO: this is not ideal, because now matronome onsets are passed just to say that there are 2
# participants. Should probably just have a condition that says how many there are
def qa_data(TapTimes, MissedCrit, TestRange=None, asyn_bound=0.5, missed_taps=False):

    # TODO: need to incorporate missed_taps

    # if there are two dimensions (i.e. Metronome + ITI), assume a single participant
    if np.ndim(TapTimes) == 2:
        two_participants = False
        TapTimes, TapTimes2 = TapTimes
    else:
        # split the pairs into separate variables so the original code works
        two_participants = True
        TapTimes, TapTimes2, metronome = TapTimes

    # asyn, ITI and IOI should be computed here
    ITI, IOI, asyn = parse_data(TapTimes, TapTimes2)

    if missed_taps:
        taps_missed = TapTimes[TapTimes == 0]
        taps_missed2 = TapTimes2[TapTimes2 == 0]

        taps_missed[0] = False
        taps_missed2[0] = False

    # if TestRange is given, assume there are synchronization+continuation phases
    # otherwise, just a single phase
    if TestRange is not None:
        start, end = TestRange
    else:
        start = 0
        end = len(asyn)

    iti = ITI

    # check if there are any asynchronies too big
    bound = IOI * asyn_bound
    large_asyn = np.logical_or(np.array(asyn) < -1 * np.array(bound), np.array(asyn) > np.array(bound))
    invalid = np.logical_or(taps_missed[start : end], large_asyn[start : end])

    n_missed_taps = sum(taps_missed[start : end])
    n_large_asyn = sum(large_asyn[start : end])
    n_invalid = sum(invalid)

    if two_participants:
        bound = ITI * asyn_bound
        large_asyn2 = np.logical_or(np.array(asyn) < -1 * np.array(bound), np.array(asyn) > np.array(bound))
        invalid2 = np.logical_or(taps_missed2[start : end], large_asyn2[start : end])

        n_missed_taps2 = sum(taps_missed2[start : end])
        n_large_asyn2 = sum(large_asyn2[start : end])
        n_invalid2 = sum(invalid2)
    # otherwise set to 0 to allow the next lines to work
    else:
        n_missed_taps2 = 0
        n_large_asyn2 = 0
        n_invalid2 = 0


    print("Checking data quality")
    print("{} large asynchronies within Test Range".format(n_large_asyn + n_large_asyn2))
    print("{} missed taps within Test Range".format(n_missed_taps + n_missed_taps2))
    print("{} total invalid rows".format(n_invalid + n_invalid2))


    if n_invalid + n_invalid2 > MissedCrit:
        print("Data rejected. Reason: too many invalid rows.")
        # might need to change these to accomodate two participants
        return TapTimes, ITI, IOI, asyn

    for i in range(len(invalid) - 1):
        if invalid[i] and invalid[i + 1] or (two_participants and invalid2[i] and invalid2[i + 1]):
            print("Data rejected. Reason: data has consecutive invalid rows.")
            return TapTimes, ITI, IOI, asyn
    
    if n_invalid > 0:
        print("Interpolating missing data using average")

        invalid_indx = [i for i, x in enumerate(invalid) if x]

        for index in invalid_indx:
            # If element is at the start or end, repeat the next / previous value 
            # ("average" of 1 value)
            if index == 0:
                asyn[index + start] = asyn[index + start + 1]
                TapTimes[index + start] = TapTimes[index + start + 1]
            elif index + start == end - 1:
                asyn[index + start] = asyn[index + start - 1]
                TapTimes[index + start] = TapTimes[index + start - 1]
            else:
                asyn[index + start] = (asyn[index + start - 1] + asyn[index + start + 1]) / 2
                TapTimes[index + start] = (TapTimes[index + start - 1] + TapTimes[index + start + 1]) / 2

            print("Interpolated trial {}".format(index + start + 1))
        
        iti = np.diff(TapTimes)
        iti = np.insert(iti, 0, 0)

    # Note: if the asynchrony is big both ways (when counting from ITI and IOI)
    # then this will run just to fix an async that has already been fixed.
    # This could be changed to account for that.
    if two_participants and n_invalid2 > 0:
        print("Interpolating missing data using average (2nd participant)")

        invalid_indx = [i for i, x in enumerate(invalid2) if x]

        for index in invalid_indx:
            # If element is at the start or end, repeat the next / previous value 
            # ("average" of 1 value)
            if index == 0:
                asyn[index + start] = asyn[index + start + 1]
                TapTimes2[index + start] = TapTimes2[index + start + 1]
            elif index + start == end - 1:
                asyn[index + start] = asyn[index + start - 1]
                TapTimes2[index + start] = TapTimes2[index + start - 1]
            else:
                asyn[index + start] = (asyn[index + start - 1] + asyn[index + start + 1]) / 2
                TapTimes2[index + start] = (TapTimes2[index + start - 1] + TapTimes2[index + start + 1]) / 2

            print("Interpolated trial {} (2nd participant)".format(index + start + 1))
        
        ioi = np.diff(TapTimes2)
        ioi = np.insert(ioi, 0, 0)


    # TODO: need to return stuff from MATLAB that's not currently here
    if two_participants:
        return (TapTimes, TapTimes2), iti, ioi, asyn
    else:
        return TapTimes, iti, IOI, asyn



# TODO: this will not exist in the final version
filename = glob.glob("*Stable*.txt")[0]

data = pd.read_csv(filename, header=None)

onsets1, onsets2 = data[3].tolist(), data[7].tolist()

# it might need to be moved to the function itself (depends on the data format)
missed_taps = np.logical_not(data[5].tolist()), np.logical_not(data[9].tolist())
test_range = [21, 71]

iti, ioi, asyn = parse_data(onsets1, onsets2)

onsets, iti, ioi, asyn = qa_data((onsets1, onsets2), 4, test_range)


# ignore the first value, since it is calculated with reference to 0
# TODO: might need to be revisited when we have data
quality_data = pd.DataFrame(list(zip(iti[(test_range[0] + 1):], asyn[(test_range[0] + 1):], ioi[(test_range[0] + 1):])))

quality_data.to_csv("cleaned_data.csv", header=None, index=None)