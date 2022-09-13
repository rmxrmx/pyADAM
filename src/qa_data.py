from typing import List
import numpy as np


# % Performs quality assurance on SMS data. Checks for invalid trials (missed taps, late (or
# % early) taps). Determines whether the trial should be rejected by checking
# % for the presence of consequtive missing taps, or if the number of missed
# % taps is equal to or greater than MissedCrit. Interpolates invalid taps if
# % the trial has not been rejected.

# TODO: this will have to be accomodated for the data that Francesca gives us

# method for getting ITI, IOI and asynchronies from onsets
# Data format: onsets1 = ITI, onsets2= IOI


def convert_to_intervals(
    onsets1: List[float], onsets2: List[float]
) -> List[List[float], List[float], List[float]]:
    iti = np.diff(onsets1)
    ioi = np.diff(onsets2)

    iti = np.insert(iti, 0, 0)
    ioi = np.insert(ioi, 0, 0)

    asyn = np.subtract(onsets1, onsets2)

    # not returning the first values because we only care about ITIs
    return iti[1:], ioi[1:], asyn[1:]


def interpolate_onsets(onsets: List[float]) -> List[float]:
    for i in enumerate(onsets):
        if np.isnan(onsets[i]):
            if i == 0:
                onsets[i] = onsets[i + 1]
            elif i == len(onsets) - 1:
                onsets[i] = onsets[-2]
            else:
                k = 1
                for j in range(i, len(onsets)):
                    if np.isnan(onsets[j]):
                        k += 1
                    else:
                        increment = (onsets[j] - onsets[i - 1]) / k
                        for h in range(i, j):
                            onsets[h] = onsets[h - 1] + increment
                        break

    return onsets


# If missed_taps is True, assume that taps with onset = None are missed
# Data should be passed such that The first tap time refers to ITI, second to IOI
# TODO: should be able to parse data if you only have ITIs IOIs asyncs
# TODO: interpolations on onsets and ITIs should be separated - code is a mess right now
def qa_data(
    tap_times: List[List[float], List[float]],
    missed_crit,
    test_range=None,
    asyn_bound=0.5,
    missed_taps=False,
    two_participants=False,
):
    """A function for performing Quality Assurance on data."""
    # clean = whether the data is good enough (below critical number of misses)
    clean = True
    interpolated = False
    n_interpolations = 0

    # split the pairs into separate variables so the original code works
    tap_times, tap_times2 = tap_times

    if missed_taps:
        taps_missed = np.isnan(tap_times)
        taps_missed2 = np.isnan(tap_times2)

        taps_missed[0] = False
        taps_missed2[0] = False
    else:
        taps_missed = [False] * len(tap_times)
        taps_missed2 = [False] * len(tap_times2)

    tap_times = interpolate_onsets(tap_times)
    tap_times2 = interpolate_onsets(tap_times2)
    # asyn, ITI and IOI should be computed here
    base_iti, base_ioi, asyn = convert_to_intervals(tap_times, tap_times2)

    # if TestRange is given, assume there are synchronization+continuation phases
    # otherwise, just a single phase
    if test_range is not None:
        start, end = test_range
    else:
        start = 0
        end = len(asyn)

    iti = base_iti
    ioi = base_ioi

    # check if there are any asynchronies too big
    bound = base_ioi * asyn_bound
    large_asyn = np.logical_or(
        np.array(asyn) < -1 * np.array(bound), np.array(asyn) > np.array(bound)
    )
    invalid = np.logical_or(taps_missed[start:end], large_asyn[start:end])

    # N.B.: n_invalid refers to the first participants invalid rows, while n_invalid2, n_missed_taps2 and
    # n_large_asyn2 refers to BOTH of the participants' invalids. If two_participants = False, then
    # n_missed_taps2, n_invalid2 and n_large_asyn2 are still used, but they refer to the SINGLE participants
    # values.
    n_invalid = sum(invalid)

    if two_participants:
        bound = iti * asyn_bound
        large_asyn2 = np.logical_or(
            np.array(asyn) < -1 * np.array(bound), np.array(asyn) > np.array(bound)
        )
        invalid2 = np.logical_or(taps_missed2[start:end], large_asyn2[start:end])

        n_missed_taps2 = sum(
            np.logical_or(taps_missed[start:end], taps_missed2[start:end])
        )
        n_large_asyn2 = sum(
            np.logical_or(large_asyn[start:end], large_asyn2[start:end])
        )
        n_invalid2 = sum(np.logical_or(invalid, invalid2))
    else:
        n_missed_taps2 = sum(taps_missed[start:end])
        n_large_asyn2 = sum(large_asyn[start:end])
        n_invalid2 = sum(invalid)

    print("Checking data quality")
    print(f"{n_large_asyn2} large asynchronies within Test Range")
    print(f"{n_missed_taps2} missed taps within Test Range")
    print(f"{n_invalid2} total invalid rows")

    if n_invalid2 > missed_crit:
        clean = False
        print("Data rejected. Reason: too many invalid rows.")
        return (
            (tap_times, tap_times2),
            base_iti,
            base_ioi,
            asyn,
            clean,
            interpolated,
            n_missed_taps2,
            n_large_asyn2,
            n_interpolations,
        )

    for i in range(len(taps_missed) - 3):
        # TODO: this is a workaround right now, should make this a parameter of the function
        if (
            taps_missed[i]
            and taps_missed[i + 1]
            and taps_missed[i + 2]
            and taps_missed[i + 3]
            or (
                two_participants
                and taps_missed2[i]
                and taps_missed2[i + 1]
                and taps_missed2[i + 2]
                and taps_missed2[i + 3]
            )
        ):
            clean = False
            print("Data rejected. Reason: data has consecutive invalid rows.")
            print("Rows: ", i, i + 1)
            return (
                (tap_times, tap_times2),
                base_iti,
                base_ioi,
                asyn,
                clean,
                interpolated,
                n_missed_taps2,
                n_large_asyn2,
                n_interpolations,
            )

    if n_invalid > 0:
        interpolated = True
        print("Interpolating missing data using average")

        invalid_indx = [i for i, x in enumerate(invalid) if x]

        for index in invalid_indx:
            n_interpolations += 1

            # If element is at the start or end, repeat the next / previous value
            # ("average" of 1 value)
            if index == 0:
                asyn[index + start] = asyn[index + start + 1]
                tap_times[index + start] = tap_times[index + start + 1]
            elif index + start == end - 1:
                asyn[index + start] = asyn[index + start - 1]
                tap_times[index + start] = tap_times[index + start - 1]
            else:
                asyn[index + start] = (
                    asyn[index + start - 1] + asyn[index + start + 1]
                ) / 2
                tap_times[index + start] = (
                    tap_times[index + start - 1] + tap_times[index + start + 1]
                ) / 2

            print(f"Interpolated trial {index + start + 1}")

        iti = np.diff(tap_times)
        iti = np.insert(iti, 0, 0)

    # N.B.: the bound is rechecked because a large asynchrony may be fixed by the part above
    # and a row that was tagged as 'invalid' would now be fine.
    if two_participants:
        bound = iti * asyn_bound
        large_asyn2 = np.logical_or(
            np.array(asyn) < -1 * np.array(bound), np.array(asyn) > np.array(bound)
        )
        invalid2 = np.logical_or(taps_missed2[start:end], large_asyn2[start:end])

        n_invalid2 = sum(invalid2)

        if n_invalid2 > 0:
            interpolated = True
            print("Interpolating missing data using average (2nd participant)")

            invalid_indx = [i for i, x in enumerate(invalid2) if x]

            for index in invalid_indx:
                n_interpolations += 1

                # If element is at the start or end, repeat the next / previous value
                # ("average" of 1 value)
                if index == 0:
                    asyn[index + start] = asyn[index + start + 1]
                    tap_times2[index + start] = tap_times2[index + start + 1]
                elif index + start == end - 1:
                    asyn[index + start] = asyn[index + start - 1]
                    tap_times2[index + start] = tap_times2[index + start - 1]
                else:
                    asyn[index + start] = (
                        asyn[index + start - 1] + asyn[index + start + 1]
                    ) / 2
                    tap_times2[index + start] = (
                        tap_times2[index + start - 1] + tap_times2[index + start + 1]
                    ) / 2

                print(f"Interpolated trial {index + start + 1} (2nd participant)")

            ioi = np.diff(tap_times2)
            ioi = np.insert(ioi, 0, 0)

    return (
        (tap_times, tap_times2),
        base_iti,
        base_ioi,
        asyn,
        clean,
        interpolated,
        n_missed_taps2,
        n_large_asyn2,
        n_interpolations,
    )


# TODO: this will not exist in the final version
# filename = glob.glob("*Stable*.txt")[0]

# data = pd.read_csv(filename, header=None)

# onsets1, onsets2 = data[3].tolist(), data[7].tolist()

# # it might need to be moved to the function itself (depends on the data format)
# missed_taps = np.logical_not(data[5].tolist()), np.logical_not(data[9].tolist())
# test_range = [21, 71]

# onsets, iti, ioi, asyn, clean, interpolated, n_missed_taps, n_large_asyn, n_interpolations = qa_data((onsets1, onsets2), 4, test_range, two_participants=True)

# # ignore the first value, since it is calculated with reference to 0
# # TODO: might need to be revisited when we have data
# quality_data = pd.DataFrame(list(zip(iti[(test_range[0] + 1):], asyn[(test_range[0] + 1):], ioi[(test_range[0] + 1):])))

# quality_data.to_csv("cleaned_data.csv", header=None, index=None)
