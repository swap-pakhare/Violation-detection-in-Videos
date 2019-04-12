import numpy as np
import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
    #from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels
    from keras_video_classifier.library.utility.ucf.UCF101_loader import scan_ucf_with_labels, scan_ucf_predict

    vgg16_include_top = True
    data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'videos')
    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    #load_ucf(data_dir_path)

    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)


    #videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()])
    videos = scan_ucf_predict(data_dir_path)
    video_file_path_list = np.array([video_file_path for video_file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0
    arr = []

    for video_file_path in video_file_path_list:
        #label = videos[video_file_path]
	video_number = video_file_path[video_file_path.rfind('/')+1: video_file_path.rfind('.')]
	video_number = int(video_number)
        predicted_label = predictor.predict(video_file_path)
        #print('predicted: ' + predicted_label + ' actual: ' + label)
        #correct_count = correct_count + 1 if label == predicted_label else correct_count
        count += 1
        #print('predicted: ' + predicted_label)
	if predicted_label == 'smoking':
	    arr.append(video_number)
	#accuracy = correct_count*100 / count
        #print('accuracy: ', accuracy)

    arr.sort(key = lambda x: x)
    print ('The timestamps are:\n')
    for num in arr:
	print (str(5*num) + ' -> ' + str(5*num+5) + '\n')

if __name__ == '__main__':
    main()
