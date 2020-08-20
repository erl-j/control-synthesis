mkdir -p ./datasets/$1/audio_16k

for i in ./datasets/$1/audio/*; do
    o=${i#rawfiles/}
    #sox "$i" -r 16000 -c 1 "${o%.raw}.wav"

    ffmpeg -i "$i" -acodec pcm_s16le -ar 16000 "$i.wav"
done