!function(e){function n(n){for(var t,o,i=n[0],c=n[1],u=0,a=[];u<i.length;u++)o=i[u],Object.prototype.hasOwnProperty.call(r,o)&&r[o]&&a.push(r[o][0]),r[o]=0;for(t in c)Object.prototype.hasOwnProperty.call(c,t)&&(e[t]=c[t]);for(f&&f(n);a.length;)a.shift()()}var t={},r={0:0};var o={};var i={3:function(){return{"./index_bg.js":{__wbindgen_object_drop_ref:function(e){return t[2].exports.z(e)},__wbindgen_cb_drop:function(e){return t[2].exports.r(e)},__wbindgen_string_new:function(e,n){return t[2].exports.C(e,n)},__wbindgen_number_new:function(e){return t[2].exports.x(e)},__wbindgen_is_falsy:function(e){return t[2].exports.u(e)},__wbg_log_3b3a07b11f9422a9:function(e,n){return t[2].exports.i(e,n)},__wbg_instanceof_Window_9c4fd26090e1d029:function(e){return t[2].exports.h(e)},__wbg_requestAnimationFrame_aa3bab1f9557a4da:function(e,n){return t[2].exports.m(e,n)},__wbg_newwithu8clampedarrayandsh_daf4b2743e8c858d:function(e,n,r,o){return t[2].exports.l(e,n,r,o)},__wbg_get_0c6963cbab34fbb6:function(e,n){return t[2].exports.d(e,n)},__wbg_call_cb478d88f3068c91:function(e,n){return t[2].exports.b(e,n)},__wbindgen_object_clone_ref:function(e){return t[2].exports.y(e)},__wbg_newnoargs_3efc7bfa69a681f9:function(e,n){return t[2].exports.k(e,n)},__wbg_call_0012cc705284c42b:function(e,n,r,o){return t[2].exports.a(e,n,r,o)},__wbg_self_05c54dcacb623b9a:function(){return t[2].exports.n()},__wbg_window_9777ce446d12989f:function(){return t[2].exports.p()},__wbg_globalThis_f0ca0bbb0149cf3d:function(){return t[2].exports.f()},__wbg_global_c3c8325ae8c7f1a9:function(){return t[2].exports.g()},__wbindgen_is_undefined:function(e){return t[2].exports.v(e)},__wbg_get_2f63a4f9b6b328d9:function(e,n){return t[2].exports.e(e,n)},__wbg_new_59cb74e423758ede:function(){return t[2].exports.j()},__wbg_stack_558ba5917b466edd:function(e,n){return t[2].exports.o(e,n)},__wbg_error_4bb6c2a97407129a:function(e,n){return t[2].exports.c(e,n)},__wbindgen_number_get:function(e,n){return t[2].exports.w(e,n)},__wbindgen_string_get:function(e,n){return t[2].exports.B(e,n)},__wbindgen_boolean_get:function(e){return t[2].exports.q(e)},__wbindgen_debug_string:function(e,n){return t[2].exports.t(e,n)},__wbindgen_throw:function(e,n){return t[2].exports.D(e,n)},__wbindgen_rethrow:function(e){return t[2].exports.A(e)},__wbindgen_closure_wrapper59:function(e,n,r){return t[2].exports.s(e,n,r)}}}}};function c(n){if(t[n])return t[n].exports;var r=t[n]={i:n,l:!1,exports:{}};return e[n].call(r.exports,r,r.exports,c),r.l=!0,r.exports}c.e=function(e){var n=[],t=r[e];if(0!==t)if(t)n.push(t[2]);else{var u=new Promise((function(n,o){t=r[e]=[n,o]}));n.push(t[2]=u);var a,s=document.createElement("script");s.charset="utf-8",s.timeout=120,c.nc&&s.setAttribute("nonce",c.nc),s.src=function(e){return c.p+""+({}[e]||e)+".js"}(e);var f=new Error;a=function(n){s.onerror=s.onload=null,clearTimeout(d);var t=r[e];if(0!==t){if(t){var o=n&&("load"===n.type?"missing":n.type),i=n&&n.target&&n.target.src;f.message="Loading chunk "+e+" failed.\n("+o+": "+i+")",f.name="ChunkLoadError",f.type=o,f.request=i,t[1](f)}r[e]=void 0}};var d=setTimeout((function(){a({type:"timeout",target:s})}),12e4);s.onerror=s.onload=a,document.head.appendChild(s)}return({1:[3]}[e]||[]).forEach((function(e){var t=o[e];if(t)n.push(t);else{var r,u=i[e](),a=fetch(c.p+""+{3:"c90c83fee5eacf6ad14c"}[e]+".module.wasm");if(u instanceof Promise&&"function"==typeof WebAssembly.compileStreaming)r=Promise.all([WebAssembly.compileStreaming(a),u]).then((function(e){return WebAssembly.instantiate(e[0],e[1])}));else if("function"==typeof WebAssembly.instantiateStreaming)r=WebAssembly.instantiateStreaming(a,u);else{r=a.then((function(e){return e.arrayBuffer()})).then((function(e){return WebAssembly.instantiate(e,u)}))}n.push(o[e]=r.then((function(n){return c.w[e]=(n.instance||n).exports})))}})),Promise.all(n)},c.m=e,c.c=t,c.d=function(e,n,t){c.o(e,n)||Object.defineProperty(e,n,{enumerable:!0,get:t})},c.r=function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},c.t=function(e,n){if(1&n&&(e=c(e)),8&n)return e;if(4&n&&"object"==typeof e&&e&&e.__esModule)return e;var t=Object.create(null);if(c.r(t),Object.defineProperty(t,"default",{enumerable:!0,value:e}),2&n&&"string"!=typeof e)for(var r in e)c.d(t,r,function(n){return e[n]}.bind(null,r));return t},c.n=function(e){var n=e&&e.__esModule?function(){return e.default}:function(){return e};return c.d(n,"a",n),n},c.o=function(e,n){return Object.prototype.hasOwnProperty.call(e,n)},c.p="",c.oe=function(e){throw console.error(e),e},c.w={};var u=window.webpackJsonp=window.webpackJsonp||[],a=u.push.bind(u);u.push=n,u=u.slice();for(var s=0;s<u.length;s++)n(u[s]);var f=a;c(c.s=0)}([function(e,n,t){t.e(1).then(t.bind(null,1)).catch(console.error).then((async function(e){const{turing:n}=e,t=document.getElementById("canvas"),r=t.getBoundingClientRect();t.style.width=2*r.width+"px",t.style.height=2*r.height+"px";var o,i=1,c=1,u=.01,a=0,s=50,f=.01,d=.75,_="Fixed",l="Fixed",b=4,p=10;const g=t.getContext("2d"),m=document.getElementById("animate"),w=[];function y(e,n,t){const r=document.getElementById(e),o=document.getElementById(n);o.innerHTML=r.value;const i=e=>{o.innerHTML=r.value,t(parseFloat(r.value))};return r.addEventListener("input",i),w.push(i),r}function h(e,n){const t=document.getElementById(e);return t.addEventListener("click",e=>{t.checked&&n(t.value)}),t}y("deltaTime","deltaTimeLabel",e=>i=e),y("skipFrames","skipFramesLabel",e=>c=e),y("visc","viscLabel",e=>u=e),y("diff","diffLabel",e=>a=e),y("density","densityLabel",e=>s=e),y("decay","decayLabel",e=>f=e),y("velo","veloLabel",e=>d=e),h("wrapX",e=>_=e),h("fixedX",e=>_=e),h("flowX",e=>_=e),h("wrapY",e=>l=e),h("fixedY",e=>l=e),h("flowY",e=>l=e),y("diffIter","diffIterLabel",e=>b=e),y("projIter","projIterLabel",e=>p=e);let x=!1;document.getElementById("buttonResetParticles").addEventListener("click",e=>{x=!0}),t.addEventListener("mousemove",e=>{o=[e.offsetX/2,e.offsetY/2]});var v=document.getElementById("label");function L(){console.time("Rendering in Rust");try{n(r.width,r.height,(e,n)=>{g.putImageData(e,0,0),v.innerHTML=n;const t={terminate:!document.getElementById("animate").checked,mousePos:o,deltaTime:i,skipFrames:c,visc:u,diff:a,density:s,decay:f,rv:d,boundaryX:_,boundaryY:l,diffIter:b,projIter:p,resetParticles:x};return x=!1,t})}catch(e){console.log("Rendering error: "+e)}console.timeEnd("Rendering in Rust")}m.onclick=e=>{m.checked&&L()},L()}))}]);