async function saveAndRedirect() {
  var selectedOption = document.querySelector('input[name="displayOption"]:checked');
  var fileInput = document.getElementById('formFileSm');
  var file = fileInput.files[0];

  if (selectedOption && file) {
    var formData = new FormData();
    formData.append('file', file);
    formData.append('selectedOption', selectedOption.value);

    // 서버로 데이터 전송
    const response = await fetch('/upload_and_process', {
      method: 'POST',
      body: formData
    });

    // 서버 응답 처리
    const result = await response.json();
    console.log(result);
  } else {
    alert('라디오 버튼을 선택하세요.');
  }
}

function showSelectedScreen() {
  // 선택된 라디오 버튼의 값을 가져옴
  var selectedOption = document.querySelector('input[name="displayOption"]:checked');

  if (selectedOption) {
    // 선택된 옵션에 따라 화면을 변경
    var resultScreen = document.getElementById('resultScreen');
    var optionResult = document.getElementById('optionResult');

    // 여기에서 선택된 옵션에 따라 다른 내용을 보여주도록 설정
    switch (selectedOption.value) {
      case 'option1':
        optionResult.textContent = '옵션 1을 선택했습니다.';
        break;
      case 'option2':
        optionResult.textContent = '옵션 2를 선택했습니다.';
        break;
      case 'option3':
        optionResult.textContent = '옵션 3을 선택했습니다.';
        break;
    }

    // 결과 화면을 보이도록 변경
    resultScreen.style.display = 'block';
  } else {
    alert('라디오 버튼을 선택하세요.');
  }
}
