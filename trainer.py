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
from misc import InfiniteSampler
from util import dense_image_warp
import os

from volumetric_rendering.ray_sampler import RaySampler
from volumetric_rendering.myrender import ImportanceRenderer
from decoder import OSGDecoder
os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7"
#torch.cuda.set_device(1)


def save_tensor(tensor, path):
    import numpy as np
    np.save(path, tensor.detach().cpu().numpy())
    print("saved")
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
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 imageFolder="/home/zyh/psp/pixel2style2pixel-master/dataset/Obama/ori_imgs",
                 crop_size=(256,256),
                 is_align=False):
        self.image_path_list = []
        self.crop_size = crop_size
        for item in sorted(os.listdir(imageFolder)):
            self.image_path_list.append(os.path.join(imageFolder, item))
    def __len__(self):
        return len(self.image_path_list)
    def __getitem__(self,idx):
        image_path = self.image_path_list[idx]
        img = io.imread(image_path)[:, :, :3]
        if self.crop_size is not None:
            img = cv2.resize(img, self.crop_size)
        img = np.array(img_as_float32(img))
        img = img.transpose((2, 0, 1))
        #img = torch.from_numpy(img).unsqueeze(0)
        return img
    def get_identity(self, idx):
        image_path = self.image_path_list[idx]
        img = io.imread(image_path)[:, :, :3]
        if self.crop_size is not None:
            img = cv2.resize(img, self.crop_size)
        img = np.array(img_as_float32(img))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        return img




def train(audio_path, img_path, model_path, save_path, audio_feature_path=None, epoch=100, mode="train"):
    
    device = torch.device("cuda", 0)
    
    # 导入或者生成音频feature
    if os.path.exists(audio_feature_path):
        audio_feature = np.load(audio_feature_path)
    else:
        audio_feature = get_audio_feature_from_audio(audio_path)
        np.save(audio_path.replace('.wav', '.npy'), audio_feature)
    
    
    config_file = "/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/config/vox-256.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f) 

    super_config = "/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/config/parameters.yaml"
    opt = argparse.Namespace(**yaml.safe_load(open(super_config)))


    #print(audio_feature.shape) # (31995, 41)
    frames = len(audio_feature) // 4  # len 7998
    MyDataset = ImageDataset() # len 8000

    training_set_sampler = InfiniteSampler(dataset=MyDataset)
    train_iterator = iter(torch.utils.data.DataLoader(dataset=MyDataset, sampler=training_set_sampler))
    #assert len(MyDataset) == frames


    identity = MyDataset.get_identity(0).to(device)

    # 对应head motion estimator 从输入图像和audio中得到head pose
    ref_pose_rot, ref_pose_trans = get_pose_from_audio(identity, audio_feature, model_path, device)
    #print("generate pose:", ref_pose_rot.shape, ref_pose_trans.shape) # (7999, 3) (7999, 3)
    #torch.cuda.empty_cache()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = MotionGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    audio2kp = AudioModel3D(opt)
    rendering_kwargs = {}
    rendering_kwargs["neural_rendering_resolution"] = (512,512)
    ray_sampler = RaySampler() # 训练参数很少，batchnorm
    renderer = ImportanceRenderer()
    render_decoder = OSGDecoder(64, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})

    grad_vars = list(kp_detector.parameters())
    grad_vars += list(generator.parameters())
    grad_vars += list(audio2kp.parameters())
    grad_vars += list(ray_sampler.parameters())
    grad_vars += list(renderer.parameters())
    grad_vars += list(render_decoder.parameters())

    kp_detector = kp_detector.to(device)  
    generator = generator.to(device)
    audio2kp = audio2kp.to(device)   
    ray_sampler = ray_sampler.to(device)
    renderer = renderer.to(device)
    render_decoder = render_decoder.to(device)

    if model_path is not None:
        print("======load motion estimator from {}======".format(model_path))
        checkpoint  = torch.load(model_path)
        kp_detector.load_state_dict(checkpoint["kp_detector"])
        generator.load_state_dict(checkpoint["generator"])
        audio2kp.load_state_dict(checkpoint["audio2kp"])
    else:
        print("training restart")
    generator.train()
    kp_detector.train()
    audio2kp.train()
    # 划分生成的audio feature和pose称为frame对应的
    # 之前的处理已经生成了head pose 和audio，这里将他们完成和帧对应的
    audio_f = []
    poses = []
    poses_vector_trans = []
    poses_vector_rot = []
    rot_mat = []
    pad = np.zeros((4,41),dtype=np.float32)
    for i in range(0, frames, opt.seq_len // 2):
        temp_audio = []
        temp_pos = []
        temp_pos_vector_trans = []
        temp_pos_vector_rot = []
        temp_rot_mat = []
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
            # 
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            #print(pose.shape)
            #print(pose.sum())
            #np.save("/home/zyh/psp/pixel2style2pixel-master/models/motion_estimator/vis/pose.npy", pose)
            temp_pos.append(pose)
            temp_pos_vector_trans.append(np.array(trans))
            temp_pos_vector_rot.append(np.array(rot))
            
            temp_mat, _ = cv2.Rodrigues(np.array(rot))
            temp_rot_mat.append(temp_mat)

        audio_f.append(temp_audio) #每次增加一个seq_len长度的feature
        poses.append(temp_pos)
        
        poses_vector_trans.append(temp_pos_vector_trans) 
        poses_vector_rot.append(temp_pos_vector_rot)
        rot_mat.append(temp_rot_mat)
        
    audio_f = torch.from_numpy(np.array(audio_f,dtype=np.float32)).unsqueeze(0).to(device)
    poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0).to(device)
    
    poses_vector_rot = torch.from_numpy(np.array(poses_vector_rot, dtype=np.float32)).unsqueeze(0).to(device)
    poses_vector_trans = torch.from_numpy(np.array(poses_vector_trans, dtype=np.float32)).unsqueeze(0).to(device)
    rot_mat = torch.from_numpy(np.array(rot_mat, dtype=np.float32)).unsqueeze(0).to(device)
    # 和draw annotation box的相机内参保持一致
    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype=np.float32)
    
    intrinsics = torch.from_numpy(camera_matrix).unsqueeze(0).to(device)
    print(poses_vector_rot.shape) # ([1, 250, 64, 3])
    print(audio_f.shape) #torch.Size([1, 250, 64, 4, 41])
    print(poses.shape) #torch.Size([1, 250, 64, 256, 256]) 
    print(rot_mat.shape) #([1, 250, 64, 3, 3])  
    #每个frame输入的pose和audio是同步根据audio确定的
    #print(audio_f[:,1].shape)
    #print(poses[:,1].shape)
    if mode == 'train':
        #optimizer = torch.optim.Adam([{"params": audio2kp.parameters(), "initial_lr": 2e-4}], lr=)
        optimizer = torch.optim.Adam(params=grad_vars, lr=1e-4, betas=(0.9, 0.999))
        mseloss = torch.nn.MSELoss(reduction='sum') 
        print("training...")
        bs = audio_f.shape[1]
        total_frame = 0
        t = {}
        for bs_idx in range(bs):
            # 每次输出seq_len = 64的feature
            print(bs_idx)
            t["audio"] = audio_f[:, bs_idx]#[1, 64, 4, 41]
            t["pose"] = poses[:, bs_idx]
            #print(t["pose"].shape, t["pose"].sum())# torch.Size([1, 64, 256, 256])
            t["id_img"] = identity
            t["rot_mat"] = rot_mat[:, bs_idx]
            t["trans"] = poses_vector_trans[:, bs_idx]
            t["rot_vec"] = poses_vector_rot[:, bs_idx]
            kp_gen_source = kp_detector(identity)
            gen_kp = audio2kp(t)
            if bs_idx == 0:
                startid = 0 # 0
                end_id = opt.seq_len // 4 * 3 #48
            else:
                startid = opt.seq_len // 4 #16
                end_id = opt.seq_len // 4 * 3 #48

            for frame_bs_idx in range(startid, end_id):
                with torch.no_grad():
                    tt = {}
                     #[1, 64, 10, 2]

                    tt["value"] = gen_kp["value"][:, frame_bs_idx]
                    if opt.estimate_jacobian:
                        tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]
                    # grandTruth image
                    gt = next(train_iterator)
                    gt = gt.to(device)
                    cur_rot_matrix = t["rot_mat"][:, frame_bs_idx]
                    cur_trans = t["trans"][:, frame_bs_idx]
                    cur_rot_vec = t["rot_vec"][:, frame_bs_idx]

                    cur_c2w_matrix = torch.eye(4)
                    cur_c2w_matrix[:3, :3] = cur_rot_matrix.squeeze(0)
                    cur_c2w_matrix[:3, 3] = cur_trans.squeeze(0)
                    
                    cur_c2w_matrix = cur_c2w_matrix.unsqueeze(0).to(device)
                    #print("---------")
                    #print(cur_c2w_matrix.shape)
                    #print(cur_c2w_matrix)

                    #print(cur_rot_vec)
                    #print(cur_rot_matrix)
                    #print(cur_trans)
                    #print(cur_rot_matrix.shape)
                    #print(cur_trans.shape)

                out_gen = generator(identity, kp_source=kp_gen_source, kp_driving=tt)
                #ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
                feature_map = out_gen["feature_map"]
                #feature_samples, depth_samples, weights_samples = renderer(feature_map, decoder, ray_origins, ray_directions, rendering_kwargs) # channels last
                print("feature map shape:", feature_map.shape)
                ray_origins, ray_directions = ray_sampler(cur_c2w_matrix, intrinsics, 64)
                print(ray_origins.shape) 
                feature_samples, depth_samples, weights_samples = renderer(feature_map, 
                                                                           render_decoder, 
                                                                           ray_origins, 
                                                                           ray_directions, 
                                                                           rendering_kwargs)
                # render过了！
                
                        # Reshape into 'raw' neural-rendered image
                H = W = rendering_kwargs["neural_rendering_resolution"]
                feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], H, W).contiguous()
                depth_image = depth_samples.permute(0, 2, 1).reshape(1, 1, H, W)

                # Run superresolution to get final image
                rgb_image = feature_image[:, :3]
                print(rgb_image.shape)
                #out_gen["kp_source"] = kp_gen_source
                #out_gen["kp_driving"] = tt
                #del out_gen['sparse_deformed']
                #del out_gen['occlusion_map']
                #del out_gen['deformed']

                #fake_batch.append(out_gen["prediction"])

                total_frame += 1            
                
                #print(gt.shape)
                #print(out_gen["prediction"].shape)
                loss = mseloss(gt, out_gen["prediction"])
                #print("train loss:", loss.data)
                #loss = calulate_loss(gt, out_gen["prediction"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    if mode == 'inference':
        print("do inference")
        bs = audio_f.shape[1] 
        predictions_gen = []
        total_frames = 0
        for bs_idx in range(bs):
            t = {}
            #截取当前frame对应的audio,pose 片段
            t["audio"] = audio_f[:, bs_idx]#.cuda()
            t["pose"] = poses[:, bs_idx]#.cuda()
            t["id_img"] = identity
            #分别对应paper中对初试帧和之后每一帧的keypoint进行检测 这里都是算的一个batch内的
            kp_gen_source = kp_detector(identity)
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
                out_gen = generator(identity, kp_source=kp_gen_source, kp_driving=tt)
                out_gen["kp_source"] = kp_gen_source
                out_gen["kp_driving"] = tt

                del out_gen['sparse_deformed']
                del out_gen['occlusion_map']
                del out_gen['deformed']
                predictions_gen.append(
                    (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))       

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
    loss = 0
    mseloss = torch.nn.MSELoss(reduction='sum') 
    #loss += img2mse(cur_frames, gt)
    loss += mseloss(cur_frames, gt)

    return loss 
def img2mse(x, y): return torch.mean((x - y) ** 2)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))



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


def draw_annotation_box(image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""
    # 相机内参
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