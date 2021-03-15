_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
optimizer_config = dict( mean_grad=True)
mean_weight = True
log_config = dict(interval=1000)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
