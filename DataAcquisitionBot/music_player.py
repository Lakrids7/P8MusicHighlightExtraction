import discord
from discord.ext import commands
import os
import asyncio

FFMPEG_OPTIONS = {
    'options': '-loglevel panic -vn'
}

MUSIC_FOLDER = "Song_data"
looping = False  # Global flag for loop mode

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

async def play_audio(ctx, track_name, loop=False):
    """Handles audio playback and looping."""
    global looping
    looping = loop  # Set loop mode

    # Ensure user is in a voice channel
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("❌ You must be in a voice channel to play music.", delete_after=10)
        return

    voice_channel = ctx.author.voice.channel

    # Find the matching song folder
    song_folder = None
    for folder in os.listdir(MUSIC_FOLDER):
        if track_name.lower() in folder.lower():
            song_folder = os.path.join(MUSIC_FOLDER, folder)
            break

    if not song_folder:
        await ctx.send(f"❌ Could not find a song matching '{track_name}'.", delete_after=10)
        return

    song_path = os.path.join(song_folder, "full_song.mp3")

    if not os.path.exists(song_path):
        await ctx.send(f"❌ File not found: {song_path}", delete_after=10)
        return

    # Connect to voice channel or move if needed
    if ctx.voice_client:
        await ctx.voice_client.move_to(voice_channel)
    else:
        await voice_channel.connect()

    voice_client = ctx.voice_client

    while True:  # Loop mode
        if not looping:
            break  # Exit loop if not in loop mode

        try:
            song_path = "Song_data/PoosayOnYourLawn.mp3" #TODO THIS IS THE LINE TO DELETE IF YOU WANT TO PLAY OTHER SONGS
            audio_source = discord.FFmpegPCMAudio(song_path, **FFMPEG_OPTIONS)
            voice_client.play(audio_source)
            await ctx.send(f"🎵 Now playing **{track_name}**{' (Looping 🔄)' if looping else ''}!", delete_after=10)
        except Exception as e:
            await ctx.send(f"❌ Error playing audio: {e}", delete_after=10)
            break

        while voice_client.is_playing():
            await asyncio.sleep(1)

    await voice_client.disconnect()


@bot.command(name="play")
async def play(ctx, *, track_name: str):
    """Plays a song once."""
    await play_audio(ctx, track_name, loop=False)


@bot.command(name="loop")
async def loop(ctx, *, track_name: str):
    """Plays a song on repeat."""
    await play_audio(ctx, track_name, loop=True)


@bot.command(name="stop")
async def stop(ctx):
    """Stops playback and disables looping."""
    global looping
    looping = False  # Disable looping

    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("🛑 Stopped playback and left the voice channel.", delete_after=10)

@bot.command(name="leave")
async def leave(ctx):
    """Forces the bot to leave the voice channel."""
    global looping
    looping = False  # Ensure loop is disabled before leaving

    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send("👋 Left the voice channel.", delete_after=10)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

