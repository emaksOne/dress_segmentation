----Installation---- 

	install dependencies
	pip3 install -r requirements.txt

	Run setup from the /externals/mask_rcnn/ directory
	python3 setup.py install

	install pycocotools



----Usage----
	
	# Train a new model starting with coco pretrained weights
    python3 dress.py train --dataset=/path/to/dress/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 dress.py train --dataset=/path/to/dress/dataset --weights=last

    # Inference. Create model with last or coco or custom specified weights. Take images from inputdir. Process it. Put result images to outputdir
    python3 dress.py inference --inputdir=/path/to/dir/with/image/to/find/mask --outputdir=/path/to/dir/where/to/put/																							processed/images 
    							--weights=(last | coco| path/to/weights)


in demo.ipynb I create two models: 
 - with resnet50 backbone
 - with resnet101 backbone

train dataset - 10 images with dress on it
val dataset - 5 images

for simplicity I consider only one class which is dress.