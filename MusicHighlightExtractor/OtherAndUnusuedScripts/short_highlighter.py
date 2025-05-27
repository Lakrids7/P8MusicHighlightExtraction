import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import librosa.feature
import numpy as np
import os
import argparse
import random
import math
import soundfile as sf # For saving wav

# --- Configuration ---
CD, HD, SR, HL, NM = 3, 30, 22050, 512, 128 # Chunk Dur, Highlight Dur, Sample Rate, Hop Length, Num Mels
TC = int(SR * CD / HL) # Time steps per chunk
if TC <= 0: TC = 1 # Ensure at least one time step
DF = 64 # Feature dim
MC = 200 # Max PE length
AH = DF * 4 # Attn/Scorer Hidden dim

# --- Data Normalize (Chunk) ---
class NC(nn.Module): # NormalizeChunk
    def forward(self, m): # m is numpy array (nm, tc)
        x = torch.tensor(m, dtype=torch.float32).unsqueeze(0) # [1, nm, tc]
        mn, std = x.mean(), x.std()
        return ((x - mn) / (std + 1e-9)).squeeze(0) # [nm, tc]

# --- Data Loader Function ---
def ld(root, hd=HD): # load_data
    data, norm = [], NC()
    for sf_ in os.listdir(root): # song_folder
        sp = os.path.join(root, sf_) # song_path
        ap = os.path.join(sp, 'full_song.mp3') # audio_path
        mp = os.path.join(sp, 'metadata.txt') # metadata_path
        if not (os.path.isfile(ap) and os.path.isfile(mp)): continue
        try:
            y, sr_o = librosa.load(ap, sr=None, mono=True) # Load original SR
            if sr_o != SR: y = librosa.resample(y=y, orig_sr=sr_o, target_sr=SR); sr=SR
            else: sr=sr_o
            dur = librosa.get_duration(y=y, sr=sr)
            mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NM, hop_length=HL), ref=np.max) # (nm, total_ts)

            hs = None # highlight_start_sec
            with open(mp, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('Detected Timestamp:'):
                        try: hs = float(line.strip().split(': ', 1)[1].split()[0]); break
                        except: hs = None; break
            if hs is None: continue
            he = hs + hd # highlight_end_sec
            gt_int = [hs, he]

            chunks, labels = [], []
            ts = mel.shape[1] # total time steps
            num_chunks = int(np.floor(ts / TC)) # Number of full chunks

            for i in range(num_chunks):
                cs = i * CD # chunk_start_sec
                ce = (i + 1) * CD # chunk_end_sec
                c_int = [cs, ce]
                overlap = max(0, min(c_int[1], gt_int[1]) - max(c_int[0], gt_int[0]))
                label = 1.0 if overlap > 0 else 0.0 # Binary label

                c_mel = mel[:, i * TC : (i + 1) * TC]
                if c_mel.shape[1] == TC: # Check for full chunk
                    chunks.append(norm(c_mel))
                    labels.append(label)

            if not chunks: continue
            data.append((chunks, torch.tensor(labels, dtype=torch.float32)))
        except: pass # Suppress errors for brevity

    print(f"Loaded {len(data)} songs.")
    return data

# --- Positional Encoding ---
class PE(nn.Module):
    def __init__(self, d_m, m_l=MC): # dim_model, max_len
        super().__init__()
        pe = torch.zeros(m_l, d_m)
        pos = torch.arange(0., m_l).unsqueeze(1)
        div = torch.exp(torch.arange(0., d_m, 2) * (-math.log(10000.0) / d_m))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, m_l, d_m]
    def forward(self, n_c): # num_chunks
        return self.pe[:, :n_c, :] # [1, n_c, d_m]

# --- Model ---
class HLModel(nn.Module):
    def __init__(self, nm=NM, tc=TC, df=DF, ml=MC, ah=AH):
        super().__init__()
        self.fe = nn.Sequential( # Feature Extractor
            nn.Conv1d(nm, df, 3, 2), nn.BatchNorm1d(df), nn.ReLU(),
            nn.Conv1d(df, df*2, 4, 2), nn.BatchNorm1d(df*2), nn.ReLU(),
            nn.Conv1d(df*2, df*4, 4, 2), nn.BatchNorm1d(df*4), nn.ReLU()
        ) # Out [B, df*4, final_ts]
        self.pe = PE(df*4, ml)
        self.cs = nn.Sequential( # Chunk Scorer
            nn.Linear(df*4, ah), nn.BatchNorm1d(ah), nn.ReLU(),
            nn.Linear(ah, ah), nn.BatchNorm1d(ah), nn.ReLU(),
            nn.Linear(ah, ah), nn.BatchNorm1d(ah), nn.Tanh(),
            nn.Linear(ah, 1), nn.Sigmoid()
        ) # Out [B, num_chunks, 1]

    def forward(self, cl): # chunks_list
        nc = len(cl) # num_chunks
        if nc == 0: return torch.empty(1, 0, 1, device=cl[0].device if cl else 'cpu') # Handle empty list

        # Feat Extraction & Pooling
        cb = torch.stack(cl, dim=0) # [nc, nm, tc]
        cf = self.fe(cb) # [nc, df*4, final_ts]
        cfp = torch.max(cf, dim=2)[0] # [nc, df*4]

        # Add PE & Score
        cfb = cfp.unsqueeze(0) # [1, nc, df*4]
        pe = self.pe(nc).to(cfb.device)
        fp = cfb + pe # [1, nc, df*4]

        scores = self.cs(fp) # [1, nc, 1]
        return scores

# --- Train Function ---
def tr(d_folder, epochs=50, lr=1e-4, s_path='model.pth'): # train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = ld(d_folder)
    if not data: print("No data."); return

    random.shuffle(data)
    td = data[:int(len(data)*0.8)] # train_data
    vd = data[int(len(data)*0.8):int(len(data)*0.9)] # val_data

    model = HLModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    bv_loss = float('inf') # best_val_loss

    print("Training...")
    for epoch in range(1, epochs + 1):
        model.train(); tl = 0. # train_loss
        for chunks, labels in td:
            labels = labels.to(device)
            scores = model([c.to(device) for c in chunks]) # [1, nc, 1]

            scores_f = scores.squeeze(0).squeeze(-1) # [nc]
            if scores_f.shape != labels.shape: continue

            loss = crit(scores_f, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()

        atl = tl / len(td) # avg_train_loss
        print(f"E{epoch}/{epochs} TrL: {atl:.4f}")

        if vd:
            model.eval(); vl = 0. # val_loss
            with torch.no_grad():
                for chunks, labels in vd:
                    labels = labels.to(device)
                    scores = model([c.to(device) for c in chunks])
                    scores_f = scores.squeeze(0).squeeze(-1)
                    if scores_f.shape != labels.shape: continue
                    vl += crit(scores_f, labels).item()
            avl = vl / len(vd) # avg_val_loss
            print(f"E{epoch}/{epochs} VL: {avl:.4f}")
            if avl < bv_loss:
                bv_loss = avl
                torch.save(model.state_dict(), s_path)
                print("Saved best")
        else:
             # No validation, save at end of epoch
             torch.save(model.state_dict(), s_path)
             print("Saved model")

    print("Training done.")

# --- Extract Function ---
def ext(a_path, m_path, hl_len=HD, device='cuda', s_wav=False): # extract, audio_path, model_path, highlight_length_sec, save_wav
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    norm = NC()
    model = HLModel().to(device)
    try: model.load_state_dict(torch.load(m_path, map_location=device))
    except: print(f"Model not found at {m_path}"); return
    model.eval()

    try:
        y, sr_o = librosa.load(a_path, sr=None, mono=True)
        if sr_o != SR: y = librosa.resample(y=y, orig_sr=sr_o, target_sr=SR); sr=SR
        else: sr=sr_o
        dur = librosa.get_duration(y=y, sr=sr)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NM, hop_length=HL), ref=np.max)

        chunks = []
        ts = mel.shape[1] # total time steps
        num_chunks = int(np.floor(ts / TC))
        for i in range(num_chunks):
            c_mel = mel[:, i * TC : (i + 1) * TC]
            if c_mel.shape[1] == TC: chunks.append(norm(c_mel))

        nc = len(chunks)
        if nc == 0: print("No chunks"); return

        with torch.no_grad():
            scores = model(chunks) # [1, nc, 1]
        c_scores = scores.squeeze(0).squeeze(-1).cpu().numpy() # [nc]

        # Sliding window on chunk scores
        ws = int(np.round(hl_len / CD)) # window_size_chunks
        if ws < 1: ws = 1
        if ws > nc: hsc, hec = 0, nc # highlight_start/end_chunk
        else:
            ws_sums = np.convolve(c_scores, np.ones(ws), mode='valid')
            hsc = np.argmax(ws_sums)
            hec = hsc + ws

        hss = hsc * CD # highlight_start_sec
        hes = min(hec * CD, dur) # highlight_end_sec

        print(f"Highlight: {hss:.2f}s to {hes:.2f}s")
        # Optional: return [hss, hes]

    except Exception as e: print(f"Error extracting: {e}"); return

    # Save Wav
    if s_wav:
        od = os.path.dirname(a_path) # output_dir
        bn = os.path.splitext(os.path.basename(a_path))[0] # base_name
        wp = os.path.join(od, f'{bn}_highlight.wav') # wav_path
        try:
            ss = int(hss * sr) # start_sample
            es = int(hes * sr) # end_sample
            es = min(es, len(y)) # Cap at audio length
            h_audio = y[ss:es] # highlight_audio
            sf.write(wp, h_audio, sr)
            print(f"Saved {wp}")
        except: print("Error saving wav")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Extract music highlight model")
    parser.add_argument('mode', choices=['train', 'extract'], help='Mode: train or extract')
    parser.add_argument('--data_folder', type=str, help='Folder for dataset (train) or audio file path (extract)')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='LR for training')
    parser.add_argument('--highlight_length', type=int, default=HD, help='Highlight length in seconds (extract)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--save_wav', action='store_true', help='Save highlight WAV (extract)')

    args = parser.parse_args()

    # Check for soundfile if saving wav
    if args.mode == 'extract' and args.save_wav:
         try: import soundfile as sf
         except ImportError: print("Install soundfile for WAV saving: pip install soundfile"); args.save_wav = False

    if args.mode == 'train':
        if not args.data_folder: parser.error("data_folder is required for training")
        tr(args.data_folder, args.epochs, args.lr, args.model_path)
    elif args.mode == 'extract':
        if not args.data_folder: parser.error("data_folder (audio file path) is required for extraction")
        ext(args.data_folder, args.model_path, args.highlight_length, args.device, args.save_wav)