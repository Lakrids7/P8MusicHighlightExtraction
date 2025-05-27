const spotifyPreviewFinder = require('spotify-preview-finder');

// Set Spotify credentials (replace with yours)
process.env.SPOTIFY_CLIENT_ID = '';
process.env.SPOTIFY_CLIENT_SECRET = '';

// Function to fetch the preview URL for a given song
async function getPreviewUrl(songName) {
    try {
        const result = await spotifyPreviewFinder(songName, 1);

        if (result.success && result.results.length > 0) {
            console.log(result.results[0].previewUrls[0] || "No preview available");
        } else {
            console.log("No preview available");
        }
    } catch (error) {
        console.error("Error:", error.message);
    }
}

// Read song name from command line argument
const songName = process.argv[2];
if (!songName) {
    console.error("Please provide a song name.");
    process.exit(1);
}

getPreviewUrl(songName);
