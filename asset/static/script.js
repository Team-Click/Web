document.addEventListener('DOMContentLoaded', function () {
    const addNameButton = document.getElementById('addNameButton');
    const searchButton = document.getElementById('searchButton');
    const collaborativeNameInput = document.getElementById('collaborativeName');
    const collaborativeList = document.getElementById('collaborativeList');
    let collaborativeNames = [];

    // 입력 버튼 클릭 시 이름 추가
    addNameButton.addEventListener('click', function () {
        const name = collaborativeNameInput.value.trim();

        // 입력된 이름이 비어 있거나 이미 리스트에 존재하면 경고 표시
        if (!name) {
            alert("이름을 입력해주세요.");
            return;
        }
        if (collaborativeNames.includes(name)) {
            alert("이미 추가된 이름입니다.");
            return;
        }

        // 이름을 리스트에 추가하고 실시간으로 업데이트
        collaborativeNames.push(name);
        updateCollaborativeList();

        // 서버에 이름 추가 요청
        fetch('/add_name', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: name })
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                alert(data.message);
            }
        });

        collaborativeNameInput.value = '';  // 입력 필드 초기화
    });

    // 협업 대상 리스트 업데이트 함수
    function updateCollaborativeList() {
        collaborativeList.innerHTML = '';
        collaborativeNames.forEach(name => {
            const listItem = document.createElement('li');
            listItem.textContent = name;
            collaborativeList.appendChild(listItem);
        });
    }

    // 검색 버튼 클릭 시 협업 시간표 요청
    searchButton.addEventListener('click', function () {
        if (collaborativeNames.length === 0) {
            alert("협업 대상 이름을 추가해주세요.");
            return;
        }

        // 협업 시간표 생성 요청 (POST 요청)
        fetch("/collaborative_timetable", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
        })
        .then(response => {
            if (response.ok) {
                window.location.href = "/collaborative_timetable";
            } else {
                alert("협업 시간표 생성에 실패했습니다.");
            }
        });
    });
});
