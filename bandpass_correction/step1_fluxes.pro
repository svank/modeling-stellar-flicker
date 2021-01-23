;---------------------------------------------------------------------------
; IDL procedure to read *.cor file from Lejeune et al. flux library, and
; output integrated (bolometric & Kepler-bandpass) fluxes
;
; Written by S. R. Cranmer, February 2020
;---------------------------------------------------------------------------

; read in Kepler bandpass data

    FILE00 = 'kepler_response_hires1.txt'

    get_lun,unit & openr,unit,FILE00 & readf,unit,nkep
    xlamA_kepband = dblarr(nkep)
    trans_kepband = dblarr(nkep)
    for i=0,nkep-1 do begin
      readf,unit,x1,x2
      xlamA_kepband(i) = x1*10.
      trans_kepband(i) = x2
    endfor
    close,unit & free_lun,unit

    trans_kepband = trans_kepband/max(trans_kepband)

; set up Lejeune arrays (specifically for the metallicity [Fe/H]=0 models)

    nlam   = 1221
    nmod   = 467
    FILE01 = 'lejeune_lcbp00.cor'

    Hnu    = dblarr(nlam,nmod)
    Flam0  = dblarr(nlam,nmod)
    xlamnm = dblarr(nlam)
    Teff   = dblarr(nmod)
    xlogg  = dblarr(nmod)
    vturb  = dblarr(nmod)
    Hmix   = dblarr(nmod)

; read in Lejeune data

    get_lun,unit & openr,unit,FILE01
    for i=0,1208,8 do begin
      readf,unit,x1,x2,x3,x4,x5,x6,x7,x8
      xlamnm(i:i+7) = [x1,x2,x3,x4,x5,x6,x7,x8]
    endfor
    readf,unit,x1,x2,x3,x4,x5
    xlamnm(1216:1220) = [x1,x2,x3,x4,x5]
    for j=0,nmod-1 do begin
      readf,unit,i1,x1,x2,x3,x4,x5
      Teff(j)  = x1
      xlogg(j) = x2
      vturb(j) = x4
      Hmix(j)  = x5
      for i=0,1211,7 do begin
        readf,unit,x1,x2,x3,x4,x5,x6,x7
        Hnu(i:i+6,j) = [x1,x2,x3,x4,x5,x6,x7]
      endfor
      readf,unit,x1,x2,x3
      Hnu(1218:1220) = [x1,x2,x3]
      if ((j mod 25) eq 0) then print,j,' / ',(nmod-1)
    endfor
    close,unit & free_lun,unit

; process Lejeune data

    xlamA0 = xlamnm * 10.0

    clight = 2.997925d17
    for j=0,nmod-1 do begin
      Flam0(*,j) = 0.4 * Hnu(*,j) * clight/xlamnm/xlamnm
    endfor

    dlamA0 = dblarr(nlam)
    for i=1,nlam-2 do begin
      dlamA0(i) = 0.5*(xlamA0(i+1)-xlamA0(i-1))
    endfor
    dlamA0(0)      = dlamA0(1)*2. - dlamA0(2)
    dlamA0(nlam-1) = dlamA0(nlam-2)*2. - dlamA0(nlam-3)

    iii_kep = where((xlamA0 ge min(xlamA_kepband)) and $
                    (xlamA0 le max(xlamA_kepband)))
    xlamA_bol  = xlamA0
    xlamA_kep  = xlamA0(iii_kep)

    dlamA_bol  = dlamA0
    dlamA_kep  = dlamA0(iii_kep)

    trans_kep = INTERPOL(trans_kepband,xlamA_kepband,xlamA_kep)

; compute bolometric & Kepler-band fluxes for each model, and output

    for j=0,nmod-1 do begin

      Flam_bol = reform(Flam0(*,j))
      Flam_kep = Flam_bol(iii_kep)

      flux_bol = TOTAL(Flam_bol*dlamA_bol)
      flux_kep = TOTAL(Flam_kep*dlamA_kep*trans_kep)

      print,Teff(j),xlogg(j),flux_bol,flux_kep, $
         format='(f9.1,f8.3,2e16.8)'

    endfor

    end
