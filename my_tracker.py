from tracker import EuclideanDistTracker
import cv2
import argparse
from engine import *
from models import build_model
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


def main(args):
    cap = cv2.VideoCapture('people2.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Video_output.mp4', fourcc, 2, (680, 720), 1)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    tracker = EuclideanDistTracker()

    device = torch.device('gpu')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model

    checkpoint = torch.load("./weights/SHTechA.pth", map_location=device)
    model.load_state_dict(checkpoint['model'])

    # convert to eval mode
    model.eval()

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while cap.isOpened():

        cv2.imwrite("image.jpg", frame1)

        width, height, _ = frame1.shape
        new_width = width // 128 * 32
        new_height = height // 128 * 32

        frame1 = cv2.resize(frame1, (new_height, new_width))

        print(frame1.shape)

        # print(dilated.shape)
        # pre-processing
        img = transform(frame1)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)

        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        detections = []
        # print(len(points))

        # DRAWING RECTANGLE BOXED
        for p in points:
            (x, y, w, h) = p[0], p[1], 5, 5
            detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)
        # print(boxes_ids)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(frame1, str(id), (int(x), int(y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame1, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        cv2.imshow('frame', frame1)
        # out.write(frame1)

        frame1 = frame2
        ret, frame2 = cap.read()

        key = cv2.waitKey(30)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
