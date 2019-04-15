rm ~/lol/demo/very_large_data/more_data/*
ffmpeg -i $1 -c copy -map 0 -segment_time 00:00:05 -f segment -reset_timestamps 1 %d.mp4
mv *.mp4 ~/lol/demo/very_large_data/more_data
python ~/lol/demo/vgg16_bidirectional_lstm_predict.py
