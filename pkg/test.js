import init, { turing } from './rd_system_wasm.js'

async function run() {
  await init()

  const canvasScale = 2.;
  const canvas = document.getElementById('canvas');
  const canvasSize = canvas.getBoundingClientRect();
  canvas.style.width = canvasSize.width * canvasScale + "px";
  canvas.style.height = canvasSize.height * canvasScale + "px";

  var deltaTime = 1.0;
  var skipFrames = 1;
  var f = 0.03;
  var k = 0.056;
  var ru = 0.07;
  var rv = 0.056;
  var mousePos;

  const ctx = canvas.getContext('2d');

  const animateCheckbox = document.getElementById("animate");
  const sliderUpdater = [];
  function sliderInit(sliderId, labelId, writer){
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    label.innerHTML = slider.value;

    const update = (_event) => {
        label.innerHTML = slider.value;
        writer(parseFloat(slider.value));
    }
    slider.addEventListener("input", update);
    sliderUpdater.push(update);
    return slider;
  }
  const deltaTimeSlider = sliderInit("deltaTime", "deltaTimeLabel", value => deltaTime = value);
  const skipFramesSlider = sliderInit("skipFrames", "skipFramesLabel", value => skipFrames = value);
  const fSlider = sliderInit("f", "fLabel", value => f = value);
  const kSlider = sliderInit("k", "kLabel", value => k = value);
  const ruSlider = sliderInit("ru", "ruLabel", value => ru = value);
  const rvSlider = sliderInit("rv", "rvLabel", value => rv = value);

  const buttonStripes = document.getElementById("buttonStripes");
  buttonStripes.addEventListener("click", (event) => {
    deltaTimeSlider.value = 1.;
    fSlider.value = 0.03;
    kSlider.value = 0.056;
    ruSlider.value = 0.09;
    rvSlider.value = 0.056;
    sliderUpdater.forEach(update => update());
  })

  const buttonWaves = document.getElementById("buttonWaves");
  buttonWaves.addEventListener("click", (event) => {
    deltaTimeSlider.value = 1.;
    fSlider.value = 0.023;
    kSlider.value = 0.052;
    ruSlider.value = 0.07;
    rvSlider.value = 0.056;
    sliderUpdater.forEach(update => update());
  })

  const buttonWavyStripes = document.getElementById("buttonWavyStripes");
  buttonWavyStripes.addEventListener("click", (event) => {
    deltaTimeSlider.value = 1.;
    fSlider.value = 0.023;
    kSlider.value = 0.052;
    ruSlider.value = 0.03;
    rvSlider.value = 0.028;
    sliderUpdater.forEach(update => update());
  })

  const buttonElastic = document.getElementById("buttonElastic");
  buttonElastic.addEventListener("click", (event) => {
    deltaTimeSlider.value = 1.;
    fSlider.value = 0.023;
    kSlider.value = 0.052;
    ruSlider.value = 0.076;
    rvSlider.value = 0.074;
    sliderUpdater.forEach(update => update());
  })

  const buttonCells = document.getElementById("buttonCells");
  buttonCells.addEventListener("click", (event) => {
    deltaTimeSlider.value = 1.;
    fSlider.value = 0.023;
    kSlider.value = 0.054;
    ruSlider.value = 0.09;
    rvSlider.value = 0.072;
    sliderUpdater.forEach(update => update());
  })

  canvas.addEventListener("mousemove", (event) => {
      mousePos = [event.offsetX / canvasScale, event.offsetY / canvasScale];
  })

  var label = document.getElementById('label');

  function startAnimation(){
    console.time('Rendering in Rust')
    try{
    //   deserialize_string(yamlText, canvasSize.width, canvasSize.height,
        turing(canvasSize.width, canvasSize.height,
            (data, average) => {
                ctx.putImageData(data, 0, 0);
                label.innerHTML = average;
                const animateCheckbox = document.getElementById("animate");
                return {
                    terminate: !animateCheckbox.checked,
                    mousePos,
                    deltaTime,
                    skipFrames,
                    f,
                    k,
                    ru,
                    rv
                };
            });
    }
    catch(e){
        console.log("Rendering error: " + e);
    }
    console.timeEnd('Rendering in Rust')
  }

  animateCheckbox.onclick = (_event) => {
      if(animateCheckbox.checked)
        startAnimation();
  }

  startAnimation();
}

run()