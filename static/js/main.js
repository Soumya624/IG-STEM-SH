(function() {
    "use strict";
  
    const select = (el, all = false) => {
      el = el.trim()
      if (all) {
        return [...document.querySelectorAll(el)]
      } else {
        return document.querySelector(el)
      }
    }
  
    const on = (type, el, listener, all = false) => {
      let selectEl = select(el, all)
      if (selectEl) {
        if (all) {
          selectEl.forEach(e => e.addEventListener(type, listener))
        } else {
          selectEl.addEventListener(type, listener)
        }
      }
    }
  
    on('click', '.scrollto', function(e) {
      if (select(this.hash)) {
        select('.nav-menu .active').classList.remove('active')
        this.parentElement.classList.toggle('active')
        toogleNav();
      }
    }, true)
  
  })()