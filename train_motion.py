import argparse
import subprocess
import python_speech_features
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import pyworld
import torch
from audio2pose import get_pose_from_audio
from skimage import io, img_as_float32
import cv2
from generator import OcclusionAwareGenerator, MotionGenerator
from keypoint_detector import KPDetector
from audio2kp import AudioModel3D
import yaml,os,imageio

from util import dense_image_warp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
#torch.cuda.set_device(1)
def get_audio_feature_from_audio(audio_path,norm = True):
    sample_rate, audio = wavfile.read(audio_path)
    print("audio sample rate:", sample_rate)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]
    if norm:
        audio = audio - np.mean(audio)
        audio = audio / np.max(np.abs(audio))
        a = python_speech_features.mfcc(audio, sample_rate)
        print(a.shape)
        b = python_speech_features.logfbank(audio, sample_rate)
        print(b.shape)
        c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
        c_flag = (c == 0.0) ^ 1
        c = inter_pitch(c, c_flag)
        c = np.expand_dims(c, axis=1)
        c_flag = np.expand_dims(c_flag, axis=1)
        frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

        cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
        return cat
### audio2head主函数



def save_tensor(tensor, path):
    import numpy as np
    np.save(path, tensor.detach().cpu().numpy())
    print("saved")








def train(audio_path, img_path, model_path, save_path, audio_feature_path=None):

    #temp_audio="./results/temp.wav"
    #command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio))
    #output = subprocess.call(command, shell=True, stdout=None)

    #audio_feature = get_audio_feature_from_audio(temp_audio)
    
    if os.path.exists(audio_feature_path):
        audio_feature = np.load(audio_feature_path)
    else:
        audio_feature = get_audio_feature_from_audio(audio_path)
        np.save(audio_path.replace('.wav', '.npy'), audio_feature)
    
    #print(audio_feature.shape) # (31995, 41)
    frames = len(audio_feature) // 4

    img = io.imread(img_path)[:, :, :3]
    img = cv2.resize(img, (256, 256))

    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    # 对应head motion estimator 从输入图像和audio中得到head pose
    ref_pose_rot, ref_pose_trans = get_pose_from_audio(img, audio_feature, model_path)
    #torch.cuda.empty_cache()
    print(ref_pose_rot.shape)

    config_file = r"./config/vox-256.yaml"
    with open(config_file) as f:
        #config = yaml.load(f)
        config = yaml.safe_load(f)
        #print(config)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    #generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
    #                                    **config['model_params']['common_params'])
    generator = MotionGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])   
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()
    super_config = "/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/config/parameters.yaml"
    opt = argparse.Namespace(**yaml.safe_load(open(super_config)))
    audio2kp = AudioModel3D(opt).cuda()

    checkpoint  = torch.load(model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    generator.load_state_dict(checkpoint["generator"])
    audio2kp.load_state_dict(checkpoint["audio2kp"])

    #generator.eval()
    #kp_detector.eval()
    #audio2kp.eval()
    # 划分生成的audio feature和pose称为frame对应的
    audio_f = []
    poses = []
    pad = np.zeros((4,41),dtype=np.float32)
    for i in range(0, frames, opt.seq_len // 2):
        temp_audio = []
        temp_pos = []
        # 下面的循环是生成每个frame对应的audio feature
        for j in range(opt.seq_len):
            if i + j < frames:
                temp_audio.append(audio_feature[(i+j)*4:(i+j)*4+4])
                trans = ref_pose_trans[i + j]
                rot = ref_pose_rot[i + j]
            else:
                temp_audio.append(pad)
                trans = ref_pose_trans[-1]
                rot = ref_pose_rot[-1]

            pose = np.zeros([256, 256])
            # 根据生成的pose画出对应的boundding box! 这里不是直接以head motion generator生成的pose作为输入
            # 而是在相机坐标系画出一个3D框表示头部姿态，并将其投影到图像中
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            temp_pos.append(pose)
        audio_f.append(temp_audio) #每次增加一个seq_len长度的feature
        poses.append(temp_pos)

    audio_f = torch.from_numpy(np.array(audio_f,dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)
    ############################### 注意对batch的理解
    bs = audio_f.shape[1] #这里是进行推理，batch的数量是audio分段的数量，这里就是根据seq_len分段的数量？！！！！！
    predictions_gen = []
    total_frames = 0
    for bs_idx in range(bs):
        t = {}
        #截取当前frame对应的audio,pose 片段
        t["audio"] = audio_f[:, bs_idx].cuda()
        t["pose"] = poses[:, bs_idx].cuda()
        t["id_img"] = img
        #分别对应paper中对初试帧和之后每一帧的keypoint进行检测 这里都是算的一个batch内的
        kp_gen_source = kp_detector(img)
        gen_kp = audio2kp(t)
        ##### 
        if bs_idx == 0:
            startid = 0
            end_id = opt.seq_len // 4 * 3
        else:
            startid = opt.seq_len // 4
            end_id = opt.seq_len // 4 * 3
        # 每次循环对应一个frame的生成 frames = len(audio_feature) // 4
        for frame_bs_idx in range(startid, end_id):
            tt = {}
            tt["value"] = gen_kp["value"][:, frame_bs_idx]
            if opt.estimate_jacobian:
                tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=tt)
            out_gen["kp_source"] = kp_gen_source
            out_gen["kp_driving"] = tt
            #print(out_gen.shape)
            #print(out_gen.keys)
            #save_tensor(out_gen['deformed'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/deformed.npy')
            #save_tensor(out_gen['deformation'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/deformation.npy')
            #save_tensor(out_gen['mask'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/mask.npy')
            #save_tensor(out_gen['sparse_deformed'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/sparse_deformed.npy')
            #save_tensor(out_gen['occlusion_map'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/occlusion_map.npy')
            #save_tensor(out_gen['jacobian'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/jacobian.npy')
            #save_tensor(out_gen['prediction'], path='/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/prediction.npy')
            del out_gen['sparse_deformed']
            del out_gen['occlusion_map']
            del out_gen['deformed']
            predictions_gen.append(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))
            

            
            
            #warped_image = warp_img(out_gen)
            #warp_img = encoder(warped_image)

            #loss = calulate_loss(warp_img)
            total_frames += 1
            if total_frames >= frames:
                break
        if total_frames >= frames:
            break

    log_dir = save_path
    if not os.path.exists(os.path.join(log_dir, "temp")):
        os.makedirs(os.path.join(log_dir, "temp"))
    image_name = os.path.basename(img_path)[:-4]+ "_" + os.path.basename(audio_path)[:-4] + ".mp4"

    video_path = os.path.join(log_dir, "temp", image_name)

    imageio.mimsave(video_path, predictions_gen, fps=25.0)

    save_video = os.path.join(log_dir, image_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)
    #os.remove(video_path)
# 图像warp
def encoder(warped_image):
    return 

def calulate_loss(cur_frames, gt):
    return 0



def inter_pitch(y,y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while(i<frame_num):
        if y_flag[i] == 0:
            while True:
                if y_flag[i]==0:
                    if i == frame_num-1:
                        if last !=-1:
                            y[last+1:] = y[last]
                        i+=1
                        break
                    i+=1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i-last+1
                fy = np.array([y[last],y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx,fy)
                fx_new = np.linspace(0,1,inter_num)
                fy_new = f(fx_new)
                y[last+1:i] = fy_new[1:-1]
                last = i
                i+=1

        else:
            last = i
            i+=1
    return y


def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    

if __name__ == "__main__":
    audio_path = "/home/zyh/psp/pixel2style2pixel-master/dataset/Obama/aud.wav"
    image_path = "/home/zyh/psp/pixel2style2pixel-master/dataset/Obama/ori_imgs/0.jpg"
    motion_model_path = "/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/pretrained_model/audio2head.pth.tar"
    save_path = "/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/pretrained_model/"
    audio_feature_path = "/home/zyh/psp/pixel2style2pixel-master/dataset/Obama/aud.npy"
    train(audio_path=audio_path, 
          img_path=image_path, 
          model_path=motion_model_path,
          save_path=save_path, 
          audio_feature_path=audio_feature_path)