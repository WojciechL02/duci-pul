import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAR'))
import torch
from PIL import Image
import demo_util
from utils.train_utils import create_pretrained_tokenizer

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'RAR')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_outputs')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    rar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAR')
    config = demo_util.get_config(os.path.join(rar_dir, 'configs', 'training', 'generator', 'rar.yaml'))
    config.experiment.generator_checkpoint = os.path.join(WEIGHTS, 'rar_xxl.bin')
    config.model.vq_model.pretrained_tokenizer_weight = os.path.join(WEIGHTS, 'maskgit-vqgan-imagenet-f16-256.bin')
    config.model.generator.hidden_size = 1408
    config.model.generator.num_hidden_layers = 40
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = 6144
    print('Loading tokenizer...')
    tokenizer = create_pretrained_tokenizer(config)
    tokenizer.to(device)
    print('Loading RAR-XXL...')
    generator = demo_util.get_rar_generator(config)
    generator.to(device)
    print('Models loaded!')
    print('Generating class 207...')
    generated_image = demo_util.sample_fn(
        generator=generator, tokenizer=tokenizer,
        labels=[207], randomize_temperature=1.02,
        guidance_scale=8.0, guidance_scale_pow=1.2, device=device)
    out_path = os.path.join(OUT_DIR, 'rar_test_class207.png')
    Image.fromarray(generated_image[0]).save(out_path)
    print(f'Saved to {out_path}')
    print('RAR test PASSED!')

if __name__ == '__main__':
    main()
