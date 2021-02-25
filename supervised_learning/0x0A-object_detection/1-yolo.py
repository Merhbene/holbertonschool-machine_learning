#!/usr/bin/env python3
""" Initialize Yolo_V3 """


class Yolo:
    """ YOLO_V3"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        f = open(classes_path, "r")
        self.class_names  = [c[:-1] for c in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        grid_height, grid_width, anchor_boxes, classes = outputs[0].shape
        classes = classes - 5
        image_height, image_width = image_size[0], image_size[1]

        dim1 = (grid_height, grid_width, anchor_boxes, 4) 
        boxes = []

        dim2 = (grid_height, grid_width, anchor_boxes, 1)
        box_confidences = []

        dim3 = (grid_height, grid_width, anchor_boxes, classes)
        box_class_probs = []

        for i in range(len(outputs)) :
            output = outputs[i]
            box = np.zeros(dim1)

            box_conf = output[:,:, :,4]
            box_conf = (1 / (1 + np.exp(-box_conf))) #segmoid
            box_conf = np.expand_dims(box_conf, axis=3) 

            box_class_p = output[:,:, :,5:]
            box_class_p = (1 / (1 + np.exp(-box_class_p))) #segmoid

            #j = i + 1 #for the scale dim
            #scale_dim = 13 * j


            #cx = [i for i in range(grid_height)]
            #cy = [i for i in range(grid_width)]

            pw = self.anchors[i,:,0] # anchor_box_width
            ph = self.anchors[i,:,1] # anchor_box_height
            # loop over cells
            for x in range(grid_height) : # x= cx , y=cy
                for y in range (grid_width):
      
                    # Center coordinates, width and height of the output
                    tx = output[y,x,:,0] 
                    ty = output[y,x,:,1] 
                    tw = output[y,x,:,2]
                    th = output[y,x,:,3]

                    bx = (1 / (1 + np.exp(-tx))) + x
                    by = (1 / (1 + np.exp(-ty))) + y


                    bw = pw * np.exp(tw)
                    bh = ph * np.exp(th)
    
                    print(pw.shape, tx.shape)


                    # Normalizing
                    bx = bx / grid_width
                    by = by / grid_height
                    bw = bw / self.model.input.shape[1].value
                    bh = bh / self.model.input.shape[2].value

                    x1 = bx - bw/2
                    y1 = by - bh/2

                    x2 = bx + bw/2
                    y2 = by + bh/2

                    #print(box[y,x,:,0].shape, x1.shape,bx.shape ,x1.shape)
                    box[y,x,:,0] = x1 * image_width
                    box[y,x,:,1] = y1 * image_height
                    box[y,x,:,2] = x2 * image_width
                    box[y,x,:,3] = y2 * image_height



            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class_p)





        return (boxes, box_confidences, box_class_probs)
