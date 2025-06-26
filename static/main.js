// Add Lenis for ultra-smooth, inertia-based scrolling with custom easing
const lenisScript = document.createElement('script');
lenisScript.src = 'https://cdn.jsdelivr.net/npm/@studio-freight/lenis@1.0.38/bundled/lenis.min.js';
document.head.appendChild(lenisScript);

lenisScript.onload = () => {
    const lenis = new window.Lenis({
        duration: 1.5, // slower for immersive feel
        easing: t => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // exponential ease-out
        smooth: true,
        smoothTouch: true,
        direction: 'vertical',
        gestureOrientation: 'vertical',
        touchMultiplier: 1.5,
        wheelMultiplier: 1.1,
        infinite: false,
        lerp: 0.08 // lower = smoother
    });

    function raf(time) {
        lenis.raf(time);
        requestAnimationFrame(raf);
    }
    requestAnimationFrame(raf);
};

// Section fade-in on scroll
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, { threshold: 0.15 });

document.querySelectorAll('.glass-card, .section').forEach(section => {
    observer.observe(section);
}); 