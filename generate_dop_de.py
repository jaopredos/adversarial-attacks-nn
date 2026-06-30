import os
import sys
import json
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CHECKPOINT_DIR   = 'dop_de_checkpoint'
OUTPUT_FILE      = 'dataset_op_de.pt'
LOG_FILE         = os.path.join(CHECKPOINT_DIR, 'progress_log.txt')
CHECKPOINT_FILE  = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
META_FILE        = os.path.join(CHECKPOINT_DIR, 'meta.json')

CHECKPOINT_EVERY = 500   # salvar checkpoint a cada N imagens

# Hiperparâmetros do DE 
DE_POPSIZE      = 400    # tamanho da população inicial
DE_MAXITER      = 400    # máximo de gerações por imagem
DE_F            = 0.5    # fator de mutação
DE_CR           = 0.75   # taxa de cruzamento (crossover)
EARLY_STOP_CONF = 0.1    # parar cedo se confiança na classe certa < 10%

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_resnet18_cifar():
    """ResNet18 adaptada para CIFAR-10 — deve ser idêntica à do notebook."""
    import torchvision.models as models
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


# ============================================================
# EVOLUÇÃO DIFERENCIAL BATCHED NA GPU
# ============================================================

def de_one_pixel_attack(model, image, true_label,
                         popsize=DE_POPSIZE, maxiter=DE_MAXITER,
                         F=DE_F, CR=DE_CR, device=DEVICE):
    """
    Ataque One-Pixel com Evolução Diferencial (DE/rand/1/bin) otimizado para GPU.

    Toda a população é avaliada em um único forward pass batched na GPU,
    garantindo speedup de ~100x em relação à avaliação serial na CPU.

    Espaço de busca: 5 dimensões por candidato
      [0] x_coord  : coordenada x do pixel (normalizada [0,1] -> [0, W-1])
      [1] y_coord  : coordenada y do pixel (normalizada [0,1] -> [0, H-1])
      [2] r_value  : canal vermelho do pixel ([0, 1])
      [3] g_value  : canal verde do pixel ([0, 1])
      [4] b_value  : canal azul do pixel ([0, 1])

    Objetivo: minimizar P(classe_correta | imagem_perturbada)
    Criterio: qualquer erro de classificacao (any-misclassification)

    Args:
        model      : ResNet18 em modo eval() na GPU
        image      : tensor [3, H, W] em CPU, valores em [0, 1]
        true_label : int — classe verdadeira da imagem
        popsize    : tamanho da populacao DE
        maxiter    : maximo de geracoes
        F          : fator de mutacao DE
        CR         : taxa de cruzamento DE
        device     : 'cuda' ou 'cpu'

    Returns:
        success    : bool — True se o ataque enganou o modelo
        adv_image  : tensor [3, H, W] — imagem adversarial (ou original se falhou)
        pred_label : int — predicao do modelo na melhor solucao encontrada
        pixel_info : dict com coordenadas e cor do pixel modificado
    """
    model.eval()
    C, H, W = image.shape
    img_gpu = image.to(device)

    # --- Inicializacao da populacao aleatoria ---
    pop     = np.random.rand(popsize, 5).astype(np.float32)   # [popsize, 5] in [0,1]
    fitness = np.ones(popsize, dtype=np.float32)

    best_fitness = 1.0
    best_pixel   = pop[0].copy()

    def apply_and_evaluate(population):
        """
        Aplica cada candidato como 1 pixel modificado e avalia a confianca
        na classe correta em batch unico na GPU.
        """
        imgs = img_gpu.unsqueeze(0).expand(popsize, -1, -1, -1).contiguous().clone()

        xs = (population[:, 0] * (W - 1)).astype(int).clip(0, W - 1)
        ys = (population[:, 1] * (H - 1)).astype(int).clip(0, H - 1)
        rs = torch.from_numpy(population[:, 2].clip(0, 1)).to(device)
        gs = torch.from_numpy(population[:, 3].clip(0, 1)).to(device)
        bs = torch.from_numpy(population[:, 4].clip(0, 1)).to(device)
        xs_t = torch.from_numpy(xs).to(device)
        ys_t = torch.from_numpy(ys).to(device)

        batch_idx = torch.arange(popsize, device=device)
        imgs[batch_idx, 0, ys_t, xs_t] = rs
        imgs[batch_idx, 1, ys_t, xs_t] = gs
        imgs[batch_idx, 2, ys_t, xs_t] = bs

        with torch.no_grad():
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            fit    = probs[:, true_label].cpu().numpy()
        return fit

    # Avaliacao inicial
    fitness = apply_and_evaluate(pop)
    min_idx = int(fitness.argmin())
    best_fitness = float(fitness[min_idx])
    best_pixel   = pop[min_idx].copy()

    idx_all = np.arange(popsize)

    # --- Loop principal DE/rand/1/bin ---
    for gen in range(maxiter):
        # Parada antecipada
        if best_fitness < EARLY_STOP_CONF:
            break

        # Geracao do trial (vetorizada)
        r1 = np.random.randint(0, popsize - 1, size=popsize)
        r2 = np.random.randint(0, popsize - 2, size=popsize)
        r3 = np.random.randint(0, popsize - 3, size=popsize)

        # Evitar colisao simples com o indice do proprio individuo
        r1 = np.where(r1 >= idx_all, r1 + 1, r1) % popsize
        r2 = np.where(r2 >= idx_all, r2 + 1, r2) % popsize
        r3 = np.where(r3 >= idx_all, r3 + 1, r3) % popsize

        mutant = (pop[r1] + F * (pop[r2] - pop[r3])).clip(0, 1)

        # Cruzamento binomial
        cross_mask = np.random.rand(popsize, 5) < CR
        rand_dim   = np.random.randint(0, 5, size=popsize)
        cross_mask[idx_all, rand_dim] = True

        trial = np.where(cross_mask, mutant, pop)

        # Avaliacao e selecao gulosa
        trial_fitness = apply_and_evaluate(trial)
        improve = trial_fitness < fitness
        pop[improve]     = trial[improve]
        fitness[improve] = trial_fitness[improve]

        # Atualizar melhor global
        min_idx = int(fitness.argmin())
        if fitness[min_idx] < best_fitness:
            best_fitness = float(fitness[min_idx])
            best_pixel   = pop[min_idx].copy()

    # --- Construir imagem adversarial final ---
    px      = best_pixel
    x_coord = int(px[0] * (W - 1))
    y_coord = int(px[1] * (H - 1))
    r_val   = float(px[2])
    g_val   = float(px[3])
    b_val   = float(px[4])

    adv_image = image.clone()
    adv_image[0, y_coord, x_coord] = r_val
    adv_image[1, y_coord, x_coord] = g_val
    adv_image[2, y_coord, x_coord] = b_val

    with torch.no_grad():
        logit = model(adv_image.unsqueeze(0).to(device))
        pred  = int(logit.argmax(dim=1).item())

    success    = (pred != true_label)
    pixel_info = {
        'x': x_coord, 'y': y_coord,
        'r': r_val, 'g': g_val, 'b': b_val,
        'best_fitness': best_fitness,
        'gens_used': gen + 1,
    }
    return success, adv_image.cpu(), pred, pixel_info


# ============================================================
# LOGGING
# ============================================================
def log(msg, also_print=True):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{timestamp}] {msg}'
    if also_print:
        print(line, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log('=' * 70)
    log('D_OP v3 — One-Pixel Attack com Evolucao Diferencial (DE)')
    log('Metodologia: Ilyas et al. (2019) + Su et al. (2019)')
    log('=' * 70)
    log(f'Device    : {DEVICE}')
    log(f'DE config : popsize={DE_POPSIZE}, maxiter={DE_MAXITER}, '
        f'F={DE_F}, CR={DE_CR}')
    log(f'Early stop: confianca na classe correta < {EARLY_STOP_CONF*100:.0f}%')
    log(f'Checkpoint: a cada {CHECKPOINT_EVERY} imagens -> {CHECKPOINT_DIR}/')
    log(f'Saida     : {OUTPUT_FILE}')
    log('')

    # Verificar se output final ja existe
    if os.path.exists(OUTPUT_FILE):
        log(f'[OK] {OUTPUT_FILE} ja existe. Nada a fazer.')
        log('     Delete o arquivo para regenerar o dataset.')
        return

    # --- Carregar modelo baseline ---
    BASELINE_PATH = 'baseline_model.pt'
    if not os.path.exists(BASELINE_PATH):
        log(f'[ERRO] {BASELINE_PATH} nao encontrado.')
        log('       Rode a Parte 1 do notebook para treinar o baseline primeiro.')
        sys.exit(1)

    log(f'Carregando {BASELINE_PATH}...')
    model = build_resnet18_cifar().to(DEVICE)
    ckpt  = torch.load(BASELINE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    log('Modelo baseline carregado.')
    log('')

    # --- Carregar CIFAR-10 trainset ---
    tf       = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=tf)
    N = len(trainset)
    log(f'CIFAR-10 trainset: {N} imagens')

    # --- Carregar checkpoint se existir ---
    start_idx   = 0
    images_list = []
    labels_list = []
    successes   = 0

    if os.path.exists(CHECKPOINT_FILE):
        log(f'[CHECKPOINT] Encontrado. Retomando...')
        ck = torch.load(CHECKPOINT_FILE, map_location='cpu', weights_only=False)
        start_idx   = ck['next_idx']
        images_list = list(ck['images'])
        labels_list = list(ck['labels'])
        successes   = ck['successes']
        log(f'[CHECKPOINT] Retomando do indice {start_idx}. '
            f'({successes} sucessos registrados)')
        log('')

    if start_idx < N:
        log(f'Processando {N - start_idx} imagens restantes...')
        log('')

        time_window = []

        for idx in range(start_idx, N):
            img, lbl = trainset[idx]
            t = (lbl + 1) % 10   # rotulo deslocado — mesma convencao do PGD

            t0 = time.time()
            success, adv_img, pred, px_info = de_one_pixel_attack(
                model, img, lbl,
                popsize=DE_POPSIZE, maxiter=DE_MAXITER,
                F=DE_F, CR=DE_CR, device=DEVICE,
            )
            elapsed = time.time() - t0

            images_list.append(adv_img if success else img)
            labels_list.append(t)
            if success:
                successes += 1

            # ETA com media movel das ultimas 200 imagens
            time_window.append(elapsed)
            if len(time_window) > 200:
                time_window.pop(0)
            avg_t   = sum(time_window) / len(time_window)
            done    = idx + 1
            rate    = successes / done * 100
            eta_s   = (N - done) * avg_t
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_s))

            if done <= 5 or done % 50 == 0:
                log(f'Img {done:5d}/{N} | Sucesso={successes:5d} ({rate:.1f}%) | '
                    f'{elapsed:.2f}s | avg={avg_t:.2f}s | ETA {eta_str}')

            # --- Checkpoint ---
            if done % CHECKPOINT_EVERY == 0:
                log(f'[CHECKPOINT] Salvando ({done}/{N})...')
                torch.save({
                    'next_idx':  done,
                    'images':    images_list,
                    'labels':    labels_list,
                    'successes': successes,
                }, CHECKPOINT_FILE)
                with open(META_FILE, 'w', encoding='utf-8') as f:
                    json.dump({
                        'next_idx':       done,
                        'total':          N,
                        'successes':      successes,
                        'success_rate_%': round(rate, 2),
                        'avg_time_s':     round(avg_t, 3),
                        'eta':            eta_str,
                        'timestamp':      time.strftime('%Y-%m-%d %H:%M:%S'),
                        'de_popsize':     DE_POPSIZE,
                        'de_maxiter':     DE_MAXITER,
                        'de_F':           DE_F,
                        'de_CR':          DE_CR,
                    }, f, indent=2)
                log(f'[CHECKPOINT] Salvo. Taxa atual: {rate:.1f}%')

    # --- Salvar dataset final ---
    log('')
    log('=' * 70)
    log('Processamento concluido! Salvando dataset final...')

    images_tensor = torch.stack(images_list, dim=0)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    final_rate    = successes / N * 100

    torch.save({
        'images':       images_tensor,
        'labels':       labels_tensor,
        'success_rate': final_rate,
        'metadata': {
            'n_images':    N,
            'n_successes': successes,
            'de_popsize':  DE_POPSIZE,
            'de_maxiter':  DE_MAXITER,
            'de_F':        DE_F,
            'de_CR':       DE_CR,
            'attack':      'one_pixel_de_any_misclassification',
            'label_rule':  't = (y + 1) % 10  (deslocado +1, mesma conv. do PGD)',
            'reference':   'Su et al. (2019) + Ilyas et al. (2019)',
            'created':     time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }, OUTPUT_FILE)

    log(f'Dataset salvo em {OUTPUT_FILE}')
    log(f'  Total                      : {N}')
    log(f'  Perturbadas com sucesso    : {successes} ({final_rate:.2f}%)')
    log(f'  Originais (DE falhou)      : {N - successes} ({100-final_rate:.2f}%)')
    log('')
    log('Proximo passo: execute a Parte 5B do notebook.')
    log('  A celula de carregamento detectara dataset_op_de.pt automaticamente.')
    log('=' * 70)

    # Limpar checkpoint apos conclusao
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log('[LIMPEZA] Arquivo de checkpoint removido.')


if __name__ == '__main__':
    main()
