#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    errors = list(map(lambda x: x[0]-x[1], zip(predictions, net_worths)))
    data = sorted(zip(ages, net_worths, errors), key = lambda x: x[2])

    cleaned_data = data[:int(len(data)*0.9)]
    print(f"len data = {len(predictions)}, len cleaned data = {len(cleaned_data)}")
    return cleaned_data
