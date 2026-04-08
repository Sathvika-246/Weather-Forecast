// Professional Weather Forecast - JavaScript

// Global function for location select change
function updateLocationDisplay(selectElement) {
    if (selectElement.value) {
        selectElement.style.color = 'var(--text-primary)';
        selectElement.style.fontWeight = '600';
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all Materialize components
    var sidenavElems = document.querySelectorAll('.sidenav');
    M.Sidenav.init(sidenavElems, {});

    // Initialize select dropdowns
    var selectElems = document.querySelectorAll('select');
    M.FormSelect.init(selectElems);

    // Initialize datepicker
    var datepickerElems = document.querySelectorAll('.datepicker');
    M.Datepicker.init(datepickerElems, {
        format: 'dd-mm-yyyy',
        autoClose: true,
        yearRange: 30,
        minDate: new Date(2000, 0, 1),
        maxDate: new Date(2030, 11, 31),
        firstDay: 1,
        i18n: {
            months: ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December'],
            monthsShort: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            weekdays: ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
            weekdaysShort: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        }
    });

    // Form submission with loading state
    var predictForm = document.getElementById('weatherForm');
    if (predictForm) {
        predictForm.addEventListener('submit', function(e) {
            var submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                var originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<span>Predicting...</span>';
                submitBtn.disabled = true;
            }
        });
    }

    // Console branding
    console.log('%c🌤️ Weather Forecast', 'font-size: 24px; font-weight: bold;');
    console.log('%c12 Cities | Advanced ML Predictions', 'font-size: 14px;');
});
