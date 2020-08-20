from revoice import unpack_voices
import matplotlib.pyplot as plt


def plot_dataset_stats(dataset, feature_frame_rate, destination_path):
    notes = get_notes(dataset, feature_frame_rate)

    velocities = [n["velocity"] for n in notes]
    pitches = [n["pitch"] for n in notes]
    duration_s = [n["duration_s"] for n in notes]

    f, (vax, pax, dax) = plt.subplots(3, 1)

    vax.hist(velocities, bins=20, color="grey")

    vax.set_xlabel("velocity value")
    vax.set_ylabel("n occurences")

    vax.set_title("velocity distribution")

    pax.hist(pitches, bins=20, color="grey")

    pax.set_xlabel("midi note nr")
    pax.set_ylabel("n occurences")

    pax.set_title("pitch distribution")

    dax.hist(duration_s, bins=20, color="grey")

    dax.set_title("duration distribution")
    dax.set_ylabel("n occurences")

    dax.set_xlabel("note duration (s)")

    f.subplots_adjust(hspace=1.5)

    f.savefig(destination_path)


def get_notes(dataset, feature_frame_rate):
    for d in dataset:
        roll = unpack_voices(d["midi_velocity"][None, ..., None].numpy().astype("int32"),
                             d["midi_pitch"][None, ..., None].numpy().astype("int32"))
        roll = roll.squeeze()
        n_pitches = roll.shape[1]
        n_timesteps = roll.shape[0]
        notes = []
        for p in range(n_pitches):
            on = False
            for t in range(n_timesteps):
                if not on and roll[t, p] > 0:
                    on = True
                    currentNote = {
                        "velocity": roll[t, p], "pitch": p, "duration_s": 1/feature_frame_rate}
                elif on:
                    if roll[t, p] < 1:
                        on = False
                        notes.append(currentNote)
                    else:
                        currentNote["duration_s"] += 1/feature_frame_rate
    return notes
