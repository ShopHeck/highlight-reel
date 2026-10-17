# Track license check: read the license from the TRACK's own license field, or the text next to its download
# link, never from page-wide badges (site footers like ccMixter's "this text and images ... CC BY-NC 4.0").
#   run.sh license_check.py <page.html|license.json|License.txt> [--business]
# Prints JSON: license, url, evidence, commercial_ok (no NC/ND), ok_for_account.
import sys,re,json,html as H
CC=re.compile(r'creativecommons\.org/(?:licenses/(by(?:-nc)?(?:-sa|-nd)?)|publicdomain/(zero|mark))/([0-9.]+)',re.I)
SITEWIDE=re.compile(r'(?i)(this (text|site|page)|text and images on this site|site content|website content|©\s*\d{4}.*(all rights|inc\b))')
def norm(m):
    kind,pd,ver=m.group(1),m.group(2),m.group(3)
    if pd=='zero':return f'CC0 {ver}'
    if pd=='mark':return f'PDM {ver}'
    return 'CC '+kind.upper()+' '+ver
def strip_sitewide(h):
    # drop footer / header / nav blocks and any element whose id/class says footer, and sentences about the site itself
    for tag in ('footer','header','nav'):h=re.sub(rf'(?is)<{tag}\b.*?</{tag}>',' ',h)
    h=re.sub(r'(?is)<div[^>]+(?:id|class)="[^"]*\bfooter\b[^"]*".*?(?=<div[^>]+(?:id|class)="[^"]*\bfooter\b|</body>|$)',' ',h)
    return h
def check_html(h):
    full=h;h=strip_sitewide(h);cands=[]
    for m in CC.finditer(h):
        ctx=h[max(0,m.start()-400):m.end()+200];txt=H.unescape(re.sub(r'<[^>]+>',' ',ctx))
        if SITEWIDE.search(txt):continue
        score=0;why=[]
        if re.search(r'rel="license"',h[max(0,m.start()-80):m.start()]):score+=2;why.append('rel=license field')
        if re.search(r'(?i)(licensed under|license:|cc:attributionName|attributionName|is licensed)',ctx):score+=2;why.append('track license text')
        dl=[d.start() for d in re.finditer(r'(?i)download',h)]
        if dl and min(abs(d-m.start()) for d in dl)<3000:score+=1;why.append('near download link')
        if score:cands.append((score,norm(m),m.group(0),why))
    sitewide=sorted({norm(m) for m in CC.finditer(full)}-{c[1] for c in cands})
    if not cands:return dict(license=None,error='no track-scoped license found',ignored_sitewide=sitewide)
    best={}
    for sc,l,u,w in cands:b=best.setdefault(l,[0,u,set()]);b[0]+=sc;b[2].update(w)
    lic=max(best,key=lambda k:best[k][0]);other=[k for k in best if k!=lic]
    return dict(license=lic,url='https://'+best[lic][1]+'/',evidence=sorted(best[lic][2]),conflicting_track_licenses=other,ignored_sitewide=sitewide)
def check_text(t):
    try:
        j=json.loads(t);flat=json.dumps(j)
        for k in ('license','licenseurl','license_url','licence'):
            v=j.get(k) if isinstance(j,dict) else None
            if v:m=CC.search(str(v));return dict(license=norm(m) if m else str(v),url=str(v),evidence=[f'json field {k}'])
        m=CC.search(flat)
    except Exception:m=CC.search(t)
    return dict(license=norm(m),url='https://'+m.group(0)+'/',evidence=['license file']) if m else dict(license=None,error='no license in file')
def check(path,business=False):
    t=open(path,errors='ignore').read();r=check_html(t) if '<html' in t[:5000].lower() or '<a ' in t else check_text(t)
    l=r.get('license') or ''
    r['commercial_ok']=bool(l) and not re.search(r'NC|ND',l)
    r['ok_for_account']=('business' if business else 'creator')+(': OK' if r['commercial_ok'] and not r.get('conflicting_track_licenses') else ': BLOCK (NC/ND/unknown/ambiguous)')
    r['file']=path;return r
if __name__=='__main__':
    b='--business' in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith('--')]:print(json.dumps(check(p,b)))
