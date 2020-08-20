import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl


def plot_prediction(target_f0_scaled, target_ld_scaled, predicted_f0_scaled, predicted_ld_scaled, midi_pitch, midi_velocity):

    mpl.rcParams['lines.linewidth'] = 4

    sns.set(rc={'figure.figsize': (20, 30)})

    sns.set_style("white")
    target_f0_scaled = target_f0_scaled.numpy().squeeze()
    target_ld_scaled = target_ld_scaled.numpy().squeeze()
    predicted_f0_scaled = predicted_f0_scaled.numpy().squeeze()
    predicted_ld_scaled = predicted_ld_scaled.numpy().squeeze()
    midi_pitch = midi_pitch.numpy().squeeze()
    midi_velocity = midi_velocity.numpy().squeeze()

    n_steps = target_f0_scaled.shape[0]

    midi_color = "orange"

    f, (mpax, pax, mlax, lax) = plt.subplots(4, 1)

    not_changing = np.abs(np.diff(midi_pitch, append=0)) == 0

    # mpax.plot(np.linspace(0,90,n_steps),alpha=0)

    mpax.fill_between(np.linspace(0, n_steps, n_steps), (midi_pitch-0.5)/127.0, (midi_pitch+0.5)/127.0, where=(midi_pitch > 0.0)*not_changing,
                      color=midi_color, alpha=0.6, label="midi")

    mpax.fill_between(np.linspace(0, n_steps, n_steps), (midi_pitch-2)/127.0, (midi_pitch+2)/127.0, where=(midi_pitch > 0.0)*not_changing,
                      color=midi_color, alpha=0.2)

    mpax.plot(predicted_f0_scaled, '--', color="black", label="predicted")

    mpax.plot(target_f0_scaled, color="grey", label="target")

    mpax2 = mpax.twinx()

    mpax2.set_yticklabels([int(np.round(y*127.0)) for y in mpax.get_yticks()])

    mpax.set_ylabel("pitch (unit scale)")

    mpax2.set_ylabel("midi note number")

    mpax.set_title("midi pitch, predicted pitch and target pitch")

    mpax.set_xlabel("timestep")

    pax.plot(predicted_f0_scaled, '--', color="black", label="predicted")

    pax.plot(target_f0_scaled, color="grey", label="target")

    pax.set_title("predicted pitch & target pitch")

    pax.set_ylabel("pitch (unit scale)")

    pax.set_xlabel("timestep")

    # velocity

    # mlax.plot(np.linspace(0,90,n_steps),alpha=0)

    not_changing = np.abs(np.diff(midi_velocity, append=0)) == 0

    mlax.fill_between(np.linspace(0, n_steps, n_steps), (midi_velocity-0.5)/127.0, (midi_velocity+0.5)/127.0, where=(midi_velocity > 0.0)*not_changing,
                      color=midi_color, alpha=0.6, label="midi")

    mlax.fill_between(np.linspace(0, n_steps, n_steps), (midi_velocity-2)/127.0, (midi_velocity+2)/127.0, where=(midi_velocity > 0.0)*not_changing,
                      color=midi_color, alpha=0.2)

    mlax.plot(predicted_ld_scaled, '--', color="black", label="predicted")

    mlax.plot(target_ld_scaled, color="grey", label="target")

    mlax.set_title("midi velocity, predicted loudness & target loudness")

    mlax2 = mlax.twinx()

    mlax2.set_yticklabels([int(np.round(y*127.0)) for y in mlax.get_yticks()])

    mlax.set_ylabel("loudness dB(A) (unit scale)")

    mlax2.set_ylabel("velocity value")

    mlax.set_xlabel("timestep")

    lax.plot(predicted_ld_scaled, '--', color="black", label="predicted")

    lax.plot(target_ld_scaled, color="grey", label="target")

    lax.set_title("predicted loudness & target loudness")

    lax.set_ylabel("loudness dB(A) (unit scale)")

    lax.set_xlabel("timestep")

    for x in [mlax, mpax, lax, pax]:
        x.legend()

    f.subplots_adjust(hspace=0.5)

    plt.show()

    return f
