import torch  #, tomesd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    AutoencoderTiny
)
from controlnet_aux.processor import Processor
from tqdm import tqdm
logging.set_verbosity_error()
import time

# from sd_single_step import StableDiffusionControlNetInpaintPipelineSingleStep, StableDiffusionInpaintPipelineSingleStep
from utils.sd_single_step import SingleStepWrapper

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, 
                 device, 
                 mode='geometry', 
                 text= '',
                 text_comp='',
                 add_directional_text= False, 
                 batch = 1, 
                 guidance_weight = 100, 
                 sds_weight_strategy = 0,
                 early_time_step_range = [0.02, 0.5],
                 late_time_step_range = [0.02, 0.5],
                 sd_version = '1.5',
                 negative_text = '',
                 negative_text_comp = '',
                 sd_model=None,
                 enable_controlnet=False,
                 control_type='op',
                 use_inpaint=False,
                 repaint=False,
                 use_legacy=False,
                 use_taesd=False):
        super().__init__()

        self.device = device
        self.mode = mode
        self.text= text
        self.text_comp = text_comp
        self.text_hand = 'a detailed photo of a hand.'
        self.add_directional_text = add_directional_text
        self.batch = batch 
        self.sd_version = sd_version
        self.use_inpaint = use_inpaint
        self.repaint = repaint
        print(f'[INFO] loading stable diffusion...')
        if self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            if self.use_inpaint and not self.repaint:
                model_key = "runwayml/stable-diffusion-inpainting"
            else:
                model_key = "runwayml/stable-diffusion-v1-5"
        if sd_model is not None:
            model_key = sd_model
        self.model_key = model_key

        self.use_legacy = use_legacy
        self.use_taesd = use_taesd

        # controlnet
        self.enable_controlnet = enable_controlnet
        controlnet = None
        if self.enable_controlnet:
            self.control_type = control_type
            if control_type == 'op':
                if sd_version == '2.1':
                    controlnet_key = "thibaud/controlnet-sd21-openpose-diffusers"
                elif sd_version == '1.5':
                    controlnet_key = "lllyasviel/sd-controlnet-openpose"
                controlnet = ControlNetModel.from_pretrained(controlnet_key).to(dtype=torch.float16, device=self.device)
                self.controlnet = controlnet

                # controlnet preprocessor
                # processor_ids = ["openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand"]
                processor_id = 'openpose_full'
                self.control_image_processor = Processor(processor_id)
            else:
                raise NotImplementedError      

        if self.use_legacy:
            if self.use_taesd:
                self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(self.device)
            else:
                self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae",torch_dtype=torch.float16).to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer",torch_dtype=torch.float16 )
            self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder",torch_dtype=torch.float32).to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet",torch_dtype=torch.float16 ).to(self.device)
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
        else:
            self.pipe_single_step = SingleStepWrapper(self.model_key, self.use_inpaint, 
                                                      self.enable_controlnet, controlnet=controlnet).to(self.device)
            self.unet.enable_xformers_memory_efficient_attention()
            # self.pipe_single_step = StableDiffusionControlNetInpaintPipelineSingleStep.from_pretrained(model_key, controlnet=self.controlnet, torch_dtype=torch.float16).to(self.device)
            # self.pipe_single_step = StableDiffusionInpaintPipelineSingleStep.from_pretrained(model_key, torch_dtype=torch.float16).to(self.device)

        if use_legacy:
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=torch.float16)
            # self.scheduler = DDIMScheduler.from_config(model_key, subfolder="scheduler", torch_dtype=torch.float16)
            # self.scheduler = DDPMScheduler.from_config(model_key, subfolder="scheduler", torch_dtype=torch.float16)
        
        if self.enable_controlnet:
            self.guess_mode = True
            controlnet_conditioning_scale = 1.0
            control_guidance_start = [0.0]
            control_guidance_end = [1.0]

            controlnet_keep = []
            timesteps = self.scheduler.timesteps
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if len(keeps) == 1 else keeps)

            if isinstance(controlnet_keep[i], list):
                self.cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                self.cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
        
        self.negative_text = negative_text
        self.negative_text_comp = negative_text_comp
        self.text_hand_z = self.get_text_embeds([self.text_hand], batch = 1)
        if add_directional_text:
            # text guidance for human space
            self.text_z = []
            self.uncond_z = []
            for d in ['front', 'side', 'back', 'side']:
                text = f"{self.text}, {d} view"
                # text = f"{d} view of {self.text}"
                negative_text = f"{self.negative_text}"
                # if d == 'back': negative_text += "face"
                text_z = self.get_text_embeds([text], batch = 1)
                uncond_z =self.get_uncond_embeds([negative_text], batch = 1)
                self.text_z.append(text_z)
                self.uncond_z.append(uncond_z)
            self.text_z = torch.cat(self.text_z)
            self.uncond_z = torch.cat(self.uncond_z)

            # text guidance for comp space
            self.text_z_comp = []
            self.uncond_z_comp = []
            for d in ['front', 'side', 'back', 'side']:
                text = f"{self.text_comp}, {d} view"
                # text = f"{d} view of {self.text}"
                negative_text = f"{self.negative_text_comp}"
                # if d == 'back': negative_text += "face"
                text_z_comp = self.get_text_embeds([text_comp], batch = 1)
                uncond_z_comp =self.get_uncond_embeds([negative_text_comp], batch = 1)
                self.text_z_comp.append(text_z_comp)
                self.uncond_z_comp.append(uncond_z_comp)
            self.text_z_comp = torch.cat(self.text_z_comp)
            self.uncond_z_comp = torch.cat(self.uncond_z_comp)
        else: 
            self.text_z = self.get_text_embeds([self.text], batch = self.batch)
            self.uncond_z =self.get_uncond_embeds([self.negative_text], batch = self.batch)
            self.text_z_comp = self.get_text_embeds([self.text_comp], batch = self.batch)
            self.uncond_z_comp =self.get_uncond_embeds([self.negative_text_comp], batch = self.batch)

        if use_legacy:
            del self.text_encoder

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step_early = int(self.num_train_timesteps * early_time_step_range[0])
        self.max_step_early = int(self.num_train_timesteps * early_time_step_range[1])
        self.min_step_late = int(self.num_train_timesteps *  late_time_step_range[0])
        self.max_step_late = int(self.num_train_timesteps *  late_time_step_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.guidance_weight = guidance_weight
        self.sds_weight_strategy = sds_weight_strategy
        print(f'[INFO] loaded stable diffusion!')

    # called when an attribute is not found:
    # comment out when using legacy code
    # def __getattr__(self, name):
    #     # assume it is implemented by self.pipe_single_step
    #     return self.pipe_single_step.__getattribute__(name)

    def get_text_embeds(self, prompt, batch=1):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        if batch > 1:
            text_embeddings = text_embeddings.repeat(batch, 1, 1)
      
        return text_embeddings
    
    def get_uncond_embeds(self, negative_prompt, batch):
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    
        if batch > 1:
            uncond_embeddings = uncond_embeddings.repeat(batch, 1, 1)
        return uncond_embeddings

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        if self.mode == 'appearance_modeling':
            
            imgs = 2 * imgs - 1

        if self.use_taesd:
            return self.vae.encode(imgs.to(torch.float16)).latents * self.vae.config.scaling_factor
        
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents
        
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    # # copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline
    # def prepare_image(
    #     self,
    #     image,
    #     width,
    #     height,
    #     batch_size,
    #     num_images_per_prompt,
    #     device,
    #     dtype,
    #     do_classifier_free_guidance=False,
    #     guess_mode=False,
    # ):
    #     if isinstance(image, list):
    #         control_images = torch.stack([transforms.ToTensor()(self.control_image_processor(control_image))
    #                                         for control_image in image])
    #     else:
    #         control_images = transforms.ToTensor()(self.control_image_processor(image, to_pil=False))
        
    #     image_batch_size = control_images.shape[0]

    #     if image_batch_size == 1:
    #         repeat_by = batch_size
    #     else:
    #         # image batch size is the same as prompt batch size
    #         repeat_by = num_images_per_prompt

    #     control_images = control_images.repeat_interleave(repeat_by, dim=0)

    #     control_images = control_images.to(device=device, dtype=dtype)

    #     if do_classifier_free_guidance and not guess_mode:
    #         control_images = torch.cat([control_images] * 2)

    #     return control_images

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def test_diffusion(self, latents, noise, num_inference_steps, text_embeddings, 
                       strength=1.0, mask=None, latents_masked=None, control_images=None):

        batch_size = latents.shape[0]

        # Prepare timesteps
        num_inference_steps_orig = self.scheduler.timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=self.device
        )

        # prepare noisy latents based on the strength
        is_strength_max = strength == 1.0

        # if strength is 1. then initialise the latents to noise, else initial to image + noise
        latents = noise if is_strength_max else self.scheduler.add_noise(latents, noise, timesteps[0:1])
        # if pure noise then scale the initial latents by the  Scheduler's init sigma
        latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                tt = torch.tensor([t] * batch_size, dtype=torch.long, device=self.device)
                noise_pred = self.forward(latents, noise, tt, text_embeddings, mask, latents_masked, control_images)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images_pil = [transforms.ToPILImage()((image / 2 + 0.5).clamp(0, 1)) for image in images]

        return images_pil    

    @torch.no_grad()
    def forward(self, latents, noise, t, text_embeddings, mask=None, latents_masked=None, control_images=None):
        ''' Assuming classifier guidance as default '''

        if self.use_inpaint and (mask is not None):
            # mask: B x 1 x H x W
            mask_inpaint = torch.nn.functional.interpolate(mask, size=latents.shape[-2:])  # B x 1 x H x W
            if latents_masked is None:
                latents_masked = latents * (mask_inpaint < 0.5)          

            # add noise
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # latents_noisy = self.scheduler.add_noise(latents_masked, noise, t)

            # concat latents, mask, masked_image_latents in the channel dimension
            num_channels_unet = self.unet.config.in_channels
            latent_model_input = self.scheduler.scale_model_input(latents_noisy, t)
            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask_inpaint, latents_masked], dim=1)
                latent_model_input = torch.cat([latent_model_input] * 2)
                tt = torch.cat([t] * 2) 
            elif num_channels_unet == 4:  # repaint scheme
                latents_masked = latents * (mask_inpaint < 0.5) 
                latents_noisy = self.scheduler.add_noise(latents_masked, noise, t)

                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                tt = torch.cat([t] * 2)
            else:
                raise NotImplementedError
        else:
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

        ''' ControlNet '''
        if self.enable_controlnet and (control_images is not None):
            # controlnet inference
            if self.guess_mode:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = text_embeddings.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = text_embeddings

            # control_images: B x 3 x H x W
            control_images = control_images.to(control_model_input.device)
            if not self.guess_mode:
                control_images = torch.cat([control_images] * 2) 
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=control_images,
                conditioning_scale=self.cond_scale,
                guess_mode=self.guess_mode,
                return_dict=False,
            )

            if self.guess_mode:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
        else:
            down_block_res_samples = None
            mid_block_res_sample = None

        noise_pred = self.unet(
                        latent_model_input,
                        tt,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_weight * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    # def forward(self, images, t, text_embeddings, mask=None, control_images=None):
    #     ''' Assumung classifier guidance as default '''

    #     image_latents = self.encode_imgs(images)
    #     noise = torch.randn_like(image_latents)
    #     latents = self.scheduler.add_noise(image_latents, noise, t)
    #     latent_model_input = self.scheduler.scale_model_input(latents, t)

    #     with torch.no_grad():
    #         ''' ControlNet '''
    #         if self.enable_controlnet and (control_images is not None):
    #             # controlnet inference
    #             if self.guess_mode:
    #                 # Infer ControlNet only for the conditional batch.
    #                 control_model_input = latents
    #                 controlnet_prompt_embeds = text_embeddings.chunk(2)[1]
    #             else:
    #                 control_model_input = latent_model_input
    #                 controlnet_prompt_embeds = text_embeddings

    #             # control_images: B x 3 x H x W
    #             control_images = control_images.to(control_model_input.device)
    #             if not self.guess_mode:
    #                 control_images = torch.cat([control_images] * 2) 
    #             down_block_res_samples, mid_block_res_sample = self.controlnet(
    #                 control_model_input,
    #                 t,
    #                 encoder_hidden_states=controlnet_prompt_embeds,
    #                 controlnet_cond=control_images,
    #                 conditioning_scale=self.cond_scale,
    #                 guess_mode=self.guess_mode,
    #                 return_dict=False,
    #             )

    #             if self.guess_mode:
    #                 # Infered ControlNet only for the conditional batch.
    #                 # To apply the output of ControlNet to both the unconditional and conditional batches,
    #                 # add 0 to the unconditional batch to keep it unchanged.
    #                 down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
    #                 mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
    #         else:
    #             down_block_res_samples = None
    #             mid_block_res_sample = None

    #         if self.use_inpaint and (mask is not None):
    #             # mask: B x 1 x H x W
    #             masked_image = images * (mask < 0.5)
    #             masked_image_latents = self.encode_imgs(masked_image)        
    #             mask_inpaint = torch.nn.functional.interpolate(mask, size=latents.shape[-2:])  # B x 1 x H x W

    #             # concat latents, mask, masked_image_latents in the channel dimension
    #             num_channels_unet = self.unet.config.in_channels
    #             if num_channels_unet == 9:
    #                 latent_model_input = torch.cat([latent_model_input, mask_inpaint, masked_image_latents], dim=1)
    #                 latent_model_input = torch.cat([latent_model_input] * 2)
    #                 tt = torch.cat([t] * 2) 
    #             else:
    #                 raise NotImplementedError
    #         else:
    #             # pred noise
    #             latent_model_input = torch.cat([latents] * 2)
    #             tt = torch.cat([t] * 2)

    #         noise_pred = self.unet(
    #                         latent_model_input,
    #                         tt,
    #                         encoder_hidden_states=text_embeddings,
    #                         down_block_additional_residuals=down_block_res_samples,
    #                         mid_block_additional_residual=mid_block_res_sample,
    #                         return_dict=False,
    #                     )[0]
    #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #         noise_pred = noise_pred_uncond + self.guidance_weight * (noise_pred_text - noise_pred_uncond)

    #     return image_latents, noise, noise_pred

    # @torch.no_grad()
    # def test_diffusion(self, images, strength, text_embeddings, image_latents=None,
    #                    mask=None, latents_masked=None, control_images=None,
    #                    num_inference_steps=50):

    #     batch_size = latents.shape[0]

    #     # Prepare timesteps
    #     num_inference_steps_orig = self.scheduler.timesteps
    #     self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    #     timesteps, num_inference_steps = self.get_timesteps(
    #         num_inference_steps=num_inference_steps, strength=strength, device=self.device
    #     )

    #     # prepare noisy latents based on the strength
    #     is_strength_max = strength == 1.0

    #     # if strength is 1. then initialise the latents to noise, else initial to image + noise
    #     latents = noise if is_strength_max else self.scheduler.add_noise(latents, noise, timesteps[0:1])
    #     # if pure noise then scale the initial latents by the  Scheduler's init sigma
    #     latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

    #     # Denoising loop
    #     num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    #     with tqdm(total=num_inference_steps) as progress_bar:
    #         for i, t in enumerate(timesteps):
    #             tt = torch.tensor([t] * batch_size, dtype=torch.long, device=self.device)
    #             noise_pred = self.forward(images, tt, text_embeddings, mask, control_images)

    #             # compute the previous noisy sample x_t -> x_t-1
    #             latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    #             # call the callback, if provided
    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()

    #     images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
    #     images_pil = [transforms.ToPILImage()((image / 2 + 0.5).clamp(0, 1)) for image in images]

    #     return images_pil


