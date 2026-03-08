const rgbText = document.querySelector(".user-class");
const adminContent = document.querySelector("#admin-button");

let hue = 0;

if (rgbText.textContent.toLowerCase() === "admin") {
    setInterval(() => {
        const color = `hsl(${hue}, 100%, 50%)`;
        rgbText.style.color = color;
        rgbText.style.textShadow = `0 0 8px ${color}, 0 0 16px ${color}`;
        hue = (hue + 2) % 360;
    }, 30);
}

else {
    adminContent.classList.add("not-admin");
}
