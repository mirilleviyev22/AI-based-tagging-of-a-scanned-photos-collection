(() => {
  let selectedAlbum = "";
  let centralList = [], currentIndex = 0, tagData = [];

  // 1) Fetch and populate the album dropdown
  function loadAlbums(){
    fetch('/api/albums')
      .then(res => res.json())
      .then(albums => {
        const sel = document.getElementById('albumSelect');
        sel.innerHTML = '';
        albums.forEach(a => {
          const opt = new Option(a, a);
          sel.add(opt);
        });
      })
      .catch(err => console.error("Failed to load albums:", err));
  }

  // 2) Create a new album on the server
  document.getElementById('createAlbumBtn').onclick = () => {
    const name = document.getElementById('newAlbumInput').value.trim();
    if (!name) return alert("Enter an album name.");
    fetch('/api/create_album', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({album: name})
    })
    .then(r => r.json())
    .then(js => {
      if (js.error) alert(js.error);
      else {
        //alert(js.message);
        loadAlbums();
      }
    });
  };

  // 3) Select an album and show the upload/process/tag sections
  document.getElementById('openAlbumBtn').onclick = () => {
    selectedAlbum = document.getElementById('albumSelect').value;
    if (!selectedAlbum) return alert("Select an album.");
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('processSection').style.display = 'block';
    document.getElementById('tagSection').style.display = 'block';
  };

  // 4) Upload user-selected files into the album
  document.getElementById('uploadBtn').onclick = () => {
    const files = document.getElementById('uploadInput').files;
    if (!files.length) return alert("Choose at least one file.");
    const form = new FormData();
    form.append('album', selectedAlbum);
    for (const f of files) form.append('files', f);
    fetch('/api/upload', {method:'POST', body: form})
      .then(r => r.json())
      .then(js => {
        if (js.error) alert(js.error);
        //else alert(js.message);
      });
  };

  // 5) Kick off the main pipeline in background
  document.getElementById('startProcessingBtn').onclick = () => {
    fetch('/api/start_processing', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({album: selectedAlbum})
    })
    .then(r => r.json())
    .then(js => {
      if (js.error) alert(js.error);
      else {
        document.getElementById('statusText').textContent = "Processing…";
        pollStatus();
      }
    });
  };

  // 6) Poll the server for processing status
  function pollStatus(){
    fetch(`/api/status/${encodeURIComponent(selectedAlbum)}`)
      .then(r => r.json())
      .then(js => {
        if (js.processing) {
          setTimeout(pollStatus, 2000);
        } else {
          document.getElementById('statusText').textContent = "Processing complete!";
        }
      })
      .catch(err => console.error("Status poll failed:", err));
  }

  // 7) Load and display the top-10 central images
  document.getElementById('loadCentralBtn').onclick = () => {
    fetch(`/api/central_images/${encodeURIComponent(selectedAlbum)}`)
      .then(r => r.json())
      .then(js => {
        if (js.error) return alert(js.error);
        centralList = js.central_images;
        if (!centralList.length) return alert("No central images found.");
        // show canvas & tagging UI
        ['centralCanvas','tagInputs','saveTagBtn','nextBtn','downloadTagsBtn']
          .forEach(id => document.getElementById(id).style.display='block');
        currentIndex = 0;
        tagData = [];
        showCurrent();
      });
  };

  // 8) Draw the current image onto the canvas
  function showCurrent(){
    const fname = centralList[currentIndex];
    const img = new Image();
    img.onload = () => {
      const c = document.getElementById('centralCanvas');
      const ctx = c.getContext('2d');
      ctx.clearRect(0,0,c.width,c.height);
      const scale = Math.min(c.width/img.width, c.height/img.height);
      ctx.drawImage(img, 0, 0, img.width*scale, img.height*scale);
    };
    img.src = `/uploads/${selectedAlbum}/${fname}`;
    // clear input fields
    ['inpPerson','inpPlace','inpTime','inpObjs']
      .forEach(id => document.getElementById(id).value = '');
  }

  // 9) Save the user’s tags for this image locally
  document.getElementById('saveTagBtn').onclick = () => {
    const fname = centralList[currentIndex];
    tagData.push({
      filename: fname,
      person:   document.getElementById('inpPerson').value,
      place:    document.getElementById('inpPlace').value,
      period:   document.getElementById('inpTime').value,
      objects:  document.getElementById('inpObjs').value
                   .split(',').map(s=>s.trim()).filter(Boolean)
    });
    alert(`Tag saved for ${fname}`);
  };

  // 10) Advance to the next central image
  document.getElementById('nextBtn').onclick = () => {
    if (currentIndex < centralList.length - 1) {
      currentIndex++;
      showCurrent();
    } else {
      alert("You've tagged all images.");
    }
  };

  // 11) POST all tags back to the server for persistence
  document.getElementById('downloadTagsBtn').onclick = () => {
    if (!tagData.length) return alert("No tags to save.");
    fetch('/api/save_tags', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ album: selectedAlbum, tags: tagData })
    })
    .then(r => r.json())
    .then(js => {
      if (js.error) alert(js.error);
     // else alert(js.message);
    });
  };

  // Initial load of album list
  loadAlbums();
})();
