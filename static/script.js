document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', function () {
        const studentId = document.getElementById('studentId').value;
        const studentName = document.getElementById('studentName').value;

        if (!studentId || !studentName) {
            alert('모든 필드를 입력해주세요.');
            return false;
        }
    });

    const searchForm = document.querySelector('form[action="{{ url_for(\'search_timetable\') }}"]');
    searchForm.addEventListener('submit', function () {
        const searchName = document.getElementById('searchName').value;

        if (!searchName) {
            alert('검색할 이름을 입력해주세요.');
            return false;
        }
    });
});
