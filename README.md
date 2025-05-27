# Extracting Music Highlights from Full Songs Using Deep Learning and Spotify Preview Data

**AAU P8 Semester Project**

Welcome to the repository for my P8 semester project at Aalborg University. This project explores the challenge of automatically identifying and extracting musical highlights from full-length songs, leveraging deep learning and Spotify's preview data.

NOTE: I will add a link to my project report when my exams are over, so you can read more about the motiviations and limitations for this project

---

## Repository Structure

This repository is organized into two main components:

### 1. `MusicHighlightExtractor`

This directory houses the core of the highlight extraction system. It includes:

*   **Model Training:** Scripts and notebooks for training the deep learning models.
*   **Model Analysis:** Code for evaluating model performance and understanding its behavior.
*   **Hyperparameter Tuning:** Scripts utilizing Optuna for optimizing model parameters.
*   **Visualization Scripts:** Tools for generating visual representations of data, model outputs, and analysis results.
*   **`OtherAndUnusedScripts/`:** A collection of auxiliary scripts. This includes:
    *   Scripts for ad-hoc data analysis or quick visualizations.
    *   Experimental code for approaches that were not pursued further (e.g., an Salami dataset-based chorus detection model that proved unsuccessful).

### 2. `DataAcquisitionBot`

This directory contains the code for a Discord bot designed to streamline the data acquisition process. Key features and components include:

*   **Discord Bot Logic:** The core Python code for running the bot.
*   **Data Processing:** Scripts for downloading and processing songs, including DTW (Dynamic Time Warping) cost computation.
*   **Usage:**
    1.  Add the bot to your Discord server/channel.
    2.  Use the following command to process a song or an entire playlist:
        ```
        !process {spotify_song_link_or_playlist_link}
        ```
        For example:
        ```
        !process https://open.spotify.com/track/your_song_id_here
        !process https://open.spotify.com/playlist/your_playlist_id_here
        ```

---

## Important Notes

### API Tokens

For security reasons, my personal **Spotify API token** and **Discord bot token** have been removed from the codebase.

To run the `DataAcquisitionBot` or any scripts requiring Spotify API access, you will need to:
1.  Obtain your own Spotify API credentials.
2.  Obtain your own Discord Bot token.
3.  Insert these tokens into the appropriate placeholder locations within the code.

The code is structured to work correctly once these tokens are provided.

### Data and Copyright

To comply with copyright laws, **all raw audio files (song data) originally used for training and analysis have been removed from this repository.**

The `DataAcquisitionBot` is provided to enable you to acquire and process your own song data for use with the `MusicHighlightExtractor` scripts. You are responsible for ensuring you have the necessary rights to use any audio data you collect.

---
