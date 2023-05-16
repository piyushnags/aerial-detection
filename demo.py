from utils import *
from models import *


def main(args: Any):
    dataset = get_dataset(args.data_dir, ann_path=args.ann_path)
    
    W, H = 320, 320

    out_path = 'out_blur.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10.0
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    resize_ = T.Resize((W, H), antialias=None)
    imgs = dataset.img_list

    for img, bbox in dataset:
        # Convert to NumPy Image format
        img = 255*img.permute(1,2,0)
        img = img.numpy().astype(np.uint8)

        # Draw the prediction from model
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        boxes = bbox['boxes']
        
        for box in boxes:
            box[2], box[3] = 1.08*box[2], 1.08*box[3]
            draw.rectangle( tuple(box), width=10, outline='red' )
        frame = np.array(pil_img)


        # Resize before saving the frame using video writer
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (W,H))
        out.write( frame )
        
    
    out.release()
    cv2.destroyAllWindows()



# Call main function
if __name__ == '__main__':
    args = parse()
    main(args)