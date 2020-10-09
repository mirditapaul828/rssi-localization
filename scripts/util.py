import matplotlib.pyplot as plt
import numpy as np


def plot_signals(signals, labels):

    """
    Auxiliary function to plot all signals.
    input:
        - signals: signals to plot
        - labels: labels of input signals
    output:
        - display plot
    """

    alphas = [1, 0.45]      # just some opacity values to facilitate visualization - 1 

    lenght = np.shape(signals)[1] #Time length for original and filtered signals - This dataset takes a RSSI measurement per 1ms unit time

    plt.figure()

    for j, sig in enumerate(signals): # iterates on every signal - original signal

        plt.plot(range(lenght), sig, '-o', label=labels[j], markersize=2, alpha=alphas[j])

    plt.grid()

    plt.ylabel('RSSI')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    return


def kalman_block(x, P, s, A, H, Q, R):

    """
    Prediction and update in Kalman filter
    input:
        - signal: signal to be filtered
        - x: previous mean state
        - P: previous variance state
        - s: current observation
        - A, H, Q, R: kalman filter parameters
    output:
        - x: mean state prediction
        - P: variance state prediction
    """

    # check laaraiedh2209 for further understand these equations
    
    x_mean = A * x + np.random.normal(0, Q, 1)
    P_mean = A * P * A + Q

    K = P_mean * H * (1 / (H * P_mean * H + R))
    x = x_mean + K * (s - H * x_mean)
    P = (1 - K * H) * P_mean

    return x, P


def kalman_filter(signal, A, H, Q, R):

    """
    Implementation of Kalman filter.
    Takes a signal and filter parameters and returns the filtered signal.
    input:
        - signal: signal to be filtered
        - A, H, Q, R: kalman filter parameters
    output:
        - filtered signal
    """

    predicted_signal = [] #List created to append

    x = signal[0]                                 # takes first value as first filter prediction
    P = 0                                         # set first covariance state value to zero
    

    predicted_signal.append(x)
    for j, s in enumerate(signal[1:]):            # iterates on the entire signal, except the first element

        x, P = kalman_block(x, P, s, A, H, Q, R)  # calculates next state prediction

        predicted_signal.append(x)                # update predicted signal with this step calculation

    return predicted_signal


