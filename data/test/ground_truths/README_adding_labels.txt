### FOR IMAGES ###
> enter inside the images/ folder
> create a .txt file with the same name as the image from which you want to get the metrics (case-sensitive)
> write in the .txt file the ground truths following the YOLO format:
<class_id> <x_center> <y_center <box_idth> <box_height>

### FOR VIDEOS ###
> enter inside the videos/ folder
> create a .txt with the same name as the video from which you want to get the metrics (case-sensitive)
> write in the .txt file the ground truths for the whole video, following the MOT format:
<frame_count> <track_id> <x_top_left_corner> <y_top_left_corner> <width> <height> <confidence> <class_id> <visibility_flag>
