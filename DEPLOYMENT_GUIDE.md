# Hugging Face Spaces Deployment Guide

## Prerequisites
1. GitHub account
2. Hugging Face account (free at https://huggingface.co/join)

## Deployment Steps

### 1. Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `instance-generator` (or your choice)
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public or Private (your choice)
3. Click "Create Space"

### 2. Upload Your Code

You have two options:

#### Option A: Upload via Web Interface
1. In your new Space, click "Files" tab
2. Click "Add file" → "Upload files"
3. Upload all files from your repository:
   - `app.py`
   - `requirements.txt`
   - `attribute_library.py`
   - `map_utils.py`
   - `retrieve_hospitals.py`
   - `retrieve_network.py`
   - All files from `REQreate/` folder
   - `.streamlit/config.toml`
4. Replace the default README.md with `README_SPACES.md` contents
5. Commit the files

#### Option B: Push from Git (Recommended)
```bash
# Add Hugging Face as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/instance-generator

# Push your code
git push hf main:main
```

Note: You may need to:
- Set up authentication: https://huggingface.co/docs/hub/security-tokens
- Rename your branch if it's not 'main': `git push hf master:main`

### 3. Wait for Build

The Space will automatically:
1. Install dependencies from requirements.txt
2. Start the Streamlit app
3. Show "Running" when ready (usually 2-3 minutes)

### 4. Test Your App

1. Click on your Space URL (e.g., `https://huggingface.co/spaces/YOUR_USERNAME/instance-generator`)
2. Test the instance generation
3. Verify download functionality works

## Important Notes

### Free Tier Limitations
- **No timeout issues**: Unlike Streamlit Cloud, Hugging Face Spaces doesn't have strict timeouts
- **CPU Basic tier**: Free, sufficient for most operations
- **Storage**: 50GB persistent storage
- **RAM**: 16GB RAM on free tier
- **Sleep**: Spaces sleep after 48 hours of inactivity, restart on first visit

### Performance Tips
1. First generation will be slower (cold start)
2. Network downloads (OSM data) take 10-15 minutes regardless of platform
3. Users can download generated files as ZIP
4. Consider pre-generating common cities if needed

### Monitoring
- View logs in the "Logs" tab of your Space
- Check for errors during generation
- Monitor resource usage in Space settings

## Updating Your Space

To update your app after deployment:

**Via Web Interface:**
1. Click "Files" tab
2. Click on file to edit
3. Make changes and commit

**Via Git:**
```bash
git push hf main:main
```

The Space will automatically rebuild and restart.

## Troubleshooting

### App won't start
- Check logs for import errors
- Verify all dependencies in requirements.txt
- Ensure all Python files are uploaded

### Out of memory
- Reduce batch sizes in generation
- Upgrade to paid tier if needed (starts at $0.05/hour)

### Long generation times
- This is normal - network downloads take 10-15 minutes
- Show progress indicators to users
- Consider implementing caching for common locations

## Support

For Hugging Face Spaces issues:
- Docs: https://huggingface.co/docs/hub/spaces
- Forum: https://discuss.huggingface.co/c/spaces/24

For REQreate issues:
- GitHub: https://github.com/michellqueiroz-ua/instance-generator
