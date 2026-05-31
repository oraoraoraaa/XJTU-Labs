function $(id){return document.getElementById(id)}
window.addEventListener('DOMContentLoaded', async ()=>{
  const sel = $('tests')
  const res = await fetch('/list_tests')
  const tests = await res.json()
  if (!Array.isArray(tests)){
    alert('Failed to list tests: '+(tests.error||JSON.stringify(tests)))
  } else {
    tests.forEach(t=>{
      const o = document.createElement('option'); o.value = t; o.textContent = t; sel.appendChild(o);
    })
  }

  $('build').addEventListener('click', async ()=>{
    $('build').disabled = true
    const res = await fetch('/build',{method:'POST'}).then(r=>r.json())
    alert(res.ok?('Built: '+res.path):('Build failed:\n'+res.error))
    $('build').disabled = false
  })

  $('run').addEventListener('click', async ()=>{
    const path = sel.value
    const srcRes = await fetch('/source?path='+encodeURIComponent(path)).then(r=>r.json())
    if(srcRes.ok){ $('source').value = srcRes.text } else { alert('Failed to load source: '+srcRes.error); return }
    const res = await fetch('/run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({path})}).then(r=>r.json())
    if(res.ok){ $('quads').value = res.quads; $('status').textContent = 'Wrote: ' + (res.out_quads || 'out.quads') } else { alert('Run failed:\n'+res.error) }
  })

  $('translate').addEventListener('click', async ()=>{
    const quads = $('quads').value
    const res = await fetch('/translate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({quads})}).then(r=>r.json())
    if(res.ok){ $('asm').value = res.asm; $('status').textContent = 'Wrote: ' + (res.out_asm || 'out.s') } else { alert('Translate failed') }
  })
})
