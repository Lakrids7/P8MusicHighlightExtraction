import discord
from discord.ext import commands
import asyncio
import music_player
import song_processing  # Importing the new module

TOKEN = ""

intents = discord.Intents.default()
intents.message_content = True  # Required for commands to work

bot = commands.Bot(command_prefix="!", intents=intents)
# ✅ Register music commands from music_player
bot.add_command(music_player.play)
bot.add_command(music_player.stop)
bot.add_command(music_player.leave)
bot.add_command(music_player.loop)

@bot.event
async def on_ready():
    print(f'✅ Logged in as {bot.user}')


@bot.command(name="process")
async def process(ctx, spotify_url: str):
    """Command to process a Spotify song or playlist link."""
    if "playlist" in spotify_url:
        await ctx.send("📂 Playlist detected! Fetching songs...", delete_after=5)
        song_links = song_processing.get_songs_from_playlist(spotify_url)

        if not song_links:
            await ctx.send("❌ Could not fetch songs from playlist.", delete_after=15)
            return

        for link in song_links:
            try:
                await ctx.send(f"🔄 Processing song: {link}", delete_after=5)
                await song_processing.process_song(ctx, link)
            except Exception as e:
                print(f"❌ Error processing song {link}: {e}")
                await ctx.send(f"⚠️ Error processing song: {link}. Skipping...", delete_after=5)
                continue  # Continue processing remaining songs even after errors

    elif "track" in spotify_url:
        await ctx.send("🔄 Processing song...", delete_after=15)
        await song_processing.process_song(ctx, spotify_url)

    else:
        await ctx.send("❌ Invalid Spotify link.", delete_after=15)



bot.run(TOKEN)