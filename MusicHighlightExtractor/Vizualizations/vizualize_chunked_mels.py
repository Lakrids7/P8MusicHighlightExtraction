import argparse, librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
SR = 22_050
HOP = 512   # Hop Length
N_FFT = 2048 # FFT window size
WINDOW_TYPE = 'hamming' # Window Type
N_MELS = 256
TARGET_N_CHUNKS = 10
# Choose a frame index reasonably far from the start/end for overlap visualization
FRAME_INDEX_TO_PLOT = 50
# Number of consecutive frames to show in the STFT -> Spectrogram plot
N_CONSECUTIVE_FRAMES = 4


def main():
    p = argparse.ArgumentParser(
        description=f"Generate Mel spec & waveform PNGs, windowing visualizations, STFT->Spectrogram plot using {WINDOW_TYPE} window for 'full_song.mp3'.") # Updated description
    p.add_argument("in_dir", help="Input folder containing 'full_song.mp3'")
    p.add_argument("out_dir", help="Output PNG directory")
    args = p.parse_args()

    f_path = Path(args.in_dir) / "full_song.mp3"
    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not f_path.is_file():
        print(f"Error: '{f_path}' not found.")
        return

    # --- Initialize paths for optional plots ---
    window_plot_path = None
    overlap_plot_path = None
    stft_plot_path = None
    stft_to_spec_plot_path = None # <<< New plot path

    try:
        print(f"Processing {f_path.name} (SR={SR}, HOP={HOP}, N_FFT={N_FFT}, Win={WINDOW_TYPE})...", end=" ", flush=True)
        y, sr_loaded = librosa.load(f_path, sr=SR, mono=True)

        if y is None or len(y) == 0:
            print("Skipped (failed to load or empty audio).")
            return

        # --- Windowing Visualizations (Frames N-1, N, N+1) ---
        # (This block remains largely unchanged)
        print("Generating windowing visualizations...", end=" ", flush=True)
        vis_error_overlap = False
        idx_prev = FRAME_INDEX_TO_PLOT - 1
        idx_curr = FRAME_INDEX_TO_PLOT
        idx_next = FRAME_INDEX_TO_PLOT + 1
        frame_indices_overlap = [idx_prev, idx_curr, idx_next]
        starts_overlap = [i * HOP for i in frame_indices_overlap]
        ends_overlap = [s + N_FFT for s in starts_overlap]

        if idx_prev < 0 or ends_overlap[-1] > len(y):
            print(f"Skipping overlap/STFT plots: Frame index {FRAME_INDEX_TO_PLOT} too close to edge.")
            vis_error_overlap = True

        if not vis_error_overlap:
            try:
                window = librosa.filters.get_window(WINDOW_TYPE, N_FFT)

                # 1. Single Frame Visualization
                audio_snippet_curr = y[starts_overlap[1]:ends_overlap[1]]
                windowed_snippet_curr = audio_snippet_curr * window
                samples_curr = np.arange(starts_overlap[1], ends_overlap[1])
                fig1, axs1 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
                # ... (plotting code for fig1 remains the same) ...
                axs1[0].plot(samples_curr, audio_snippet_curr)
                axs1[0].set_title(f"Original Audio Signal (Samples {starts_overlap[1]}-{ends_overlap[1]-1})")
                axs1[0].set_ylabel("Amplitude"); axs1[0].grid(True, linestyle=':', alpha=0.6); axs1[0].margins(x=0.01)
                axs1[1].plot(samples_curr, window)
                axs1[1].set_title(f"{WINDOW_TYPE.capitalize()} Window (Length {N_FFT})")
                axs1[1].set_ylabel("Weight"); axs1[1].grid(True, linestyle=':', alpha=0.6); axs1[1].margins(x=0.01)
                axs1[2].plot(samples_curr, windowed_snippet_curr)
                axs1[2].set_title("Signal After Applying Window")
                axs1[2].set_ylabel("Amplitude"); axs1[2].set_xlabel(f"Sample Index (Relative to start of song at {SR} Hz)")
                axs1[2].grid(True, linestyle=':', alpha=0.6); axs1[2].margins(x=0.01)
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                window_plot_path = out_p / f"{f_path.stem}_windowing_frame_{idx_curr}_{WINDOW_TYPE}.png"
                fig1.savefig(window_plot_path, dpi=150); plt.close(fig1)
                print(f"Saved single frame plot.", end=" ", flush=True)


                # 2. Overlapping Frames Visualization
                audio_snippets = [y[s:e] for s, e in zip(starts_overlap, ends_overlap)]
                windowed_snippets = [snip * window for snip in audio_snippets]
                sample_indices = [np.arange(s, e) for s, e in zip(starts_overlap, ends_overlap)]
                colors = plt.get_cmap('tab10').colors[:3]
                fig2, axs2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                # ... (plotting code for fig2 remains the same, using gray top plot, synced axes) ...
                neutral_color = 'dimgray' # Top Plot: Original Segments (gray)
                for i, snip in enumerate(audio_snippets):
                    axs2[0].plot(sample_indices[i], snip, color=neutral_color, alpha=0.8)
                axs2[0].set_title("Overlapping Original Audio Segments")
                axs2[0].set_ylabel("Amplitude"); axs2[0].grid(True, linestyle=':', alpha=0.6); axs2[0].margins(x=0.01)
                for i in range(3): # Middle plot: Overlapping Windows
                    axs2[1].plot(sample_indices[i], window, label=f'Window {frame_indices_overlap[i]}', color=colors[i], linestyle='--', linewidth=1.5)
                axs2[1].set_title(f"Overlapping {WINDOW_TYPE.capitalize()} Windows (Hop={HOP}, Length={N_FFT})")
                axs2[1].set_ylabel("Weight"); axs2[1].grid(True, linestyle=':', alpha=0.6); axs2[1].legend(fontsize='small'); axs2[1].margins(x=0.01)
                for i, w_snip in enumerate(windowed_snippets): # Bottom plot: Windowed Segments
                    axs2[2].plot(sample_indices[i], w_snip, label=f'Frame {frame_indices_overlap[i]}', color=colors[i], alpha=0.9)
                axs2[2].set_title("Windowed Audio Segments")
                axs2[2].set_ylabel("Amplitude"); axs2[2].set_xlabel(f"Sample Index (Relative to start of song at {SR} Hz)")
                axs2[2].grid(True, linestyle=':', alpha=0.6); axs2[2].legend(fontsize='small'); axs2[2].margins(x=0.01)
                ymin0, ymax0 = axs2[0].get_ylim(); ymin2, ymax2 = axs2[2].get_ylim() # Sync Y-Axes
                final_ymin = min(ymin0, ymin2); final_ymax = max(ymax0, ymax2)
                y_margin = (final_ymax - final_ymin) * 0.05
                final_ymin -= y_margin; final_ymax += y_margin
                axs2[0].set_ylim(final_ymin, final_ymax); axs2[2].set_ylim(final_ymin, final_ymax)
                min_sample = starts_overlap[0]; max_sample = ends_overlap[-1] # Set shared x-limits
                axs2[2].set_xlim(min_sample - HOP, max_sample + HOP)
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                overlap_plot_path = out_p / f"{f_path.stem}_overlapping_windows_{idx_prev}-{idx_next}_{WINDOW_TYPE}.png"
                fig2.savefig(overlap_plot_path, dpi=150); plt.close(fig2)
                print(f"Saved overlap plot.", end=" ", flush=True)


                # 3. STFT Magnitude of the Center Windowed Frame
                # (This plot also remains)
                stft_result_single = np.fft.fft(windowed_snippet_curr)
                magnitude_single = np.abs(stft_result_single)
                freqs_single = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
                magnitude_db_single = librosa.amplitude_to_db(magnitude_single, ref=np.max)
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(freqs_single, magnitude_db_single[:len(freqs_single)])
                ax3.set_title(f"Magnitude Spectrum (STFT) of {WINDOW_TYPE.capitalize()}-Windowed Frame {idx_curr}")
                ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Magnitude (dB)")
                ax3.grid(True, linestyle=':', alpha=0.6)
                ax3.set_ylim(np.max(magnitude_db_single)-80, np.max(magnitude_db_single)+5)
                ax3.set_xlim(0, SR / 2); plt.tight_layout()
                stft_plot_path = out_p / f"{f_path.stem}_stft_magnitude_frame_{idx_curr}_{WINDOW_TYPE}.png"
                fig3.savefig(stft_plot_path, dpi=150); plt.close(fig3)
                print(f"Saved STFT magnitude plot.", end=" ", flush=True)

            except Exception as plot_e:
                print(f"Failed during windowing visualization plot generation: {plot_e}", end=" ", flush=True)
                vis_error_overlap = True # Mark error if any visualization failed
        # --- End Windowing Visualizations ---

        # --- NEW: STFT -> Spectrogram Visualization ---
        print("Generating STFT->Spectrogram visualization...", end=" ", flush=True)
        vis_error_stft_spec = False
        try:
            # Calculate full STFT
            D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window=WINDOW_TYPE, center=True)
            D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            n_total_frames = D_db.shape[1]

            # Select consecutive frame indices for plotting individual spectra
            start_frame_idx = FRAME_INDEX_TO_PLOT # Use the same central frame index
            # Ensure we don't go past the end
            end_frame_idx = min(start_frame_idx + N_CONSECUTIVE_FRAMES, n_total_frames)
            selected_indices = np.arange(start_frame_idx, end_frame_idx)

            if len(selected_indices) < 2: # Need at least 2 frames to show progression
                 print(f"Skipping STFT->Spec plot: Not enough frames available from index {start_frame_idx}.")
                 vis_error_stft_spec = True
            else:
                # Get frequencies and times for the full STFT
                freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
                times = librosa.frames_to_time(np.arange(n_total_frames), sr=SR, hop_length=HOP, n_fft=N_FFT)

                # Create the plot (2 rows, Spectrogram larger)
                fig4, axs4 = plt.subplots(2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [1, 3]})
                fig4.suptitle('STFT Frames to Spectrogram', fontsize=14)
                colors_stft = plt.get_cmap('viridis')(np.linspace(0, 0.9, len(selected_indices))) # Use a different colormap

                # Panel 1: Plot individual STFT magnitude spectra
                for i, frame_idx in enumerate(selected_indices):
                    stft_vector = D_db[:, frame_idx]
                    axs4[0].plot(freqs, stft_vector, color=colors_stft[i], alpha=0.8, label=f'Frame {frame_idx}')
                axs4[0].set_title(f'Individual STFT Magnitude Spectra (Frames {selected_indices[0]}-{selected_indices[-1]})')
                axs4[0].set_ylabel('Magnitude (dB)')
                axs4[0].set_xlabel('Frequency (Hz)')
                axs4[0].set_xlim(0, SR/2)
                axs4[0].legend(fontsize='small')
                axs4[0].grid(True, linestyle=':', alpha=0.6)

                # Panel 2: Plot full STFT Spectrogram
                img = librosa.display.specshow(D_db, sr=SR, hop_length=HOP, x_axis='time', y_axis='linear',
                                               ax=axs4[1], cmap='magma', n_fft=N_FFT) # Pass n_fft for correct time mapping with center=True
                axs4[1].set_title('Full STFT Spectrogram (Magnitude dB)')
                # Add colorbar
                cbar = fig4.colorbar(img, ax=axs4[1], format="%+2.0f dB")
                cbar.set_label('Magnitude (dB)')


                # Highlight the selected frames on the spectrogram
                ymin_spec, ymax_spec = axs4[1].get_ylim() # Get freq axis limits
                for i, frame_idx in enumerate(selected_indices):
                    t = times[frame_idx]
                    # Draw vertical line
                    axs4[1].axvline(t, color=colors_stft[i], linestyle='--', alpha=0.7, linewidth=1.5)
                    # Add text label at the bottom
                    axs4[1].text(t, ymin_spec + (ymax_spec - ymin_spec) * 0.03, # Position slightly above bottom
                                 f'{frame_idx}', color='white', # White text stands out on magma
                                 ha='center', va='bottom', fontsize='small', fontweight='bold',
                                 bbox=dict(facecolor='black', alpha=0.4, pad=0.1, boxstyle='round,pad=0.1'))

                plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
                stft_to_spec_plot_path = out_p / f"{f_path.stem}_stft_to_spectrogram_{WINDOW_TYPE}.png"
                fig4.savefig(stft_to_spec_plot_path, dpi=150)
                plt.close(fig4)
                print(f"Saved STFT->Spectrogram plot.", end=" ", flush=True)

        except Exception as plot_e:
            print(f"Failed during STFT->Spectrogram plot generation: {plot_e}", end=" ", flush=True)
            vis_error_stft_spec = True
        # --- End STFT -> Spectrogram Visualization ---


        # --- Mel Spectrogram Calculation and Chunking ---
        print("Calculating Mel Spectrogram...", end=" ", flush=True)
        # Use the specified window type in melspectrogram
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS, window=WINDOW_TYPE) # Use WINDOW_TYPE
        m_db_t = librosa.power_to_db(S, ref=np.max).T
        n_frames = m_db_t.shape[0]
        chunk_frames = 0; n_chunks = 0; first_4_chunks_data = []; waveform_saved = False

        # (Rest of the Mel/Chunking/Saving logic is unchanged)
        if TARGET_N_CHUNKS <= 0: print("Skipped Mel Spec (TARGET_N_CHUNKS must be positive).")
        elif n_frames == 0: print("Skipped Mel Spec (n_frames is zero after melspectrogram).")
        else:
            chunk_frames = n_frames // TARGET_N_CHUNKS
            if chunk_frames == 0:
                print("Skipped Mel Spec chunks (too short to create target chunks).")
                if 'm_db_t' in locals() and m_db_t.size > 0:
                     plt.imsave(out_p / f"{f_path.stem}_full.png", np.flipud(m_db_t.T), cmap='magma')
            else:
                plt.imsave(out_p / f"{f_path.stem}_full.png", np.flipud(m_db_t.T), cmap='magma')
                n_chunks = n_frames // chunk_frames
                print(f"Generating {n_chunks} chunk spectrograms...", end=" ", flush=True)
                for i in range(n_chunks):
                    start, end = i * chunk_frames, min((i + 1) * chunk_frames, n_frames)
                    if start >= end: continue
                    chunk = m_db_t[start:end].T
                    if chunk.size > 0:
                        plt.imsave(out_p / f"{f_path.stem}_chunk_{i:03d}.png", np.flipud(chunk), cmap='magma')
                        if i < 4: first_4_chunks_data.append(chunk)
                    else: print(f"Warning: Chunk {i} was empty, skipped saving.")

                if len(first_4_chunks_data) == 4:
                    if all(c.shape[0] == N_MELS and c.shape[1] > 0 for c in first_4_chunks_data):
                        print("Generating combined spec and waveform...", end=" ", flush=True)
                        try:
                            combined = np.hstack(first_4_chunks_data)
                            plt.imsave(out_p / f"{f_path.stem}_first_4_chunks.png", np.flipud(combined), cmap='magma')
                        except ValueError as hstack_err: print(f" Failed to combine first 4 chunks: {hstack_err}.")

                        num_frames_first_4 = 4 * chunk_frames
                        end_sample_index = min(librosa.frames_to_samples(num_frames_first_4, hop_length=HOP, n_fft=N_FFT), len(y))
                        audio_segment = y[0:end_sample_index]
                        if len(audio_segment) > 0:
                            fig_wf, ax_wf = plt.subplots(figsize=(12, 2))
                            librosa.display.waveshow(audio_segment, sr=SR, ax=ax_wf, color='white', linewidth=0.5)
                            ax_wf.axis('off'); ax_wf.margins(0, 0.1)
                            fig_wf.patch.set_facecolor('black'); ax_wf.set_facecolor('black')
                            fig_wf.tight_layout(pad=0)
                            waveform_path = out_p / f"{f_path.stem}_first_4_chunks_waveform.png"
                            fig_wf.savefig(waveform_path, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='black')
                            plt.close(fig_wf); waveform_saved = True
                        else: print(" Note: Waveform segment for first 4 chunks was empty or too short.")
                    else: print(" Note: First 4 chunks have inconsistent shapes, skipping combined spec.")
                elif n_chunks >= 4 : print(f" Note: Only {len(first_4_chunks_data)} non-empty chunks collected for combining (needed 4).")


        # --- Final Summary ---
        saved_png_count = 0; summary_parts = []
        # Check for specific window type filenames
        window_plot_check_path = out_p / f"{f_path.stem}_windowing_frame_{idx_curr}_{WINDOW_TYPE}.png"
        overlap_plot_check_path = out_p / f"{f_path.stem}_overlapping_windows_{idx_prev}-{idx_next}_{WINDOW_TYPE}.png"
        stft_plot_check_path = out_p / f"{f_path.stem}_stft_magnitude_frame_{idx_curr}_{WINDOW_TYPE}.png"
        stft_to_spec_check_path = out_p / f"{f_path.stem}_stft_to_spectrogram_{WINDOW_TYPE}.png" # <<< Check new plot

        if window_plot_check_path.exists(): saved_png_count += 1; summary_parts.append("1 single-frame window plot")
        if overlap_plot_check_path.exists(): saved_png_count += 1; summary_parts.append("1 overlap window plot")
        if stft_plot_check_path.exists(): saved_png_count += 1; summary_parts.append("1 STFT magnitude plot")
        if stft_to_spec_check_path.exists(): saved_png_count += 1; summary_parts.append("1 STFT->Spectrogram plot") # <<< Add new plot to summary

        # (Rest of summary checks remain the same)
        full_spec_path = out_p / f"{f_path.stem}_full.png"
        if full_spec_path.exists(): saved_png_count += 1; summary_parts.append("1 full spec")
        actual_chunks_saved = len(list(out_p.glob(f"{f_path.stem}_chunk_*.png")))
        if actual_chunks_saved > 0: saved_png_count += actual_chunks_saved; summary_parts.append(f"{actual_chunks_saved} chunks")
        combined_spec_path = out_p / f"{f_path.stem}_first_4_chunks.png"
        if combined_spec_path.exists(): saved_png_count += 1; summary_parts.append("1 combined spec")
        waveform_check_path = out_p / f"{f_path.stem}_first_4_chunks_waveform.png"
        if waveform_check_path.exists(): saved_png_count += 1; summary_parts.append("1 waveform")

        if saved_png_count > 0 :
             # Check all possible error flags for newline
             if vis_error_overlap or vis_error_stft_spec or chunk_frames >= 0 or TARGET_N_CHUNKS <=0 or n_frames == 0: print()
             print(f"Done. Saved {saved_png_count} PNGs ({', '.join(summary_parts)}).")
        else: print("\nDone. No output PNGs were generated.")

    except Exception as e:
        print(f"\nCritical Error during processing of {f_path.name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()